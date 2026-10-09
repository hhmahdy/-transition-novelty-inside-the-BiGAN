#!/usr/bin/env python3
"""A protocol-accurate stand-in for the Unity Noisy-TV player.

The original environment is a Unity application that the ``unityagents`` client
launches as ``tv_maze[.x86_64] --port <port>`` and then drives over a TCP
socket with a JSON-plus-image-bytes protocol. The real executable is
distributed out of band (Google Drive link in the README of
``luchris429/noisy-tv-env``), so it cannot always be present - for example on a
machine with no access to Google Drive.

This script implements the *player side of that protocol* exactly as the
vendored client expects it:

1. connect to the port given by ``--port`` (the client binds it first and then
   launches the player),
2. send the academy/brain-parameters JSON and wait for the client's ``.``
   acknowledgment,
3. answer ``RESET`` (acknowledge, read the reset-parameter JSON, send the first
   state), ``STEP`` (acknowledge, read the action JSON, send the next state),
   and ``EXIT``,
4. send one JPEG/PNG frame per camera resolution per agent, confirming each
   with the client's ``RECEIVED`` handshake, followed by the global-done flag.

The socket is treated as a *byte stream*: the client may coalesce its
acknowledgment and the following command into a single TCP segment, so every
message is parsed out of a buffer rather than assumed to be one ``recv``.

The simulated maze is a straight corridor with the upstream reward (``+1``
within 2.5 units of the goal). When the episode's ``tv`` parameter is ``1.0``
the television redraws a random pattern every step - the noisy-TV condition -
and when it is ``0.0`` the frame is frozen. That is enough to drive the training
pipeline through the *real* client and the *real* adapter and to assert the
noisy-television phenomenon end to end without the Unity binary.

Usage (the client does this automatically):
    fake_tv_maze_player.py --port 5005
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import sys
from typing import Any, Iterable, Optional

import numpy as np

try:
    from PIL import Image
except ImportError as error:  # pragma: no cover - pillow is a client dependency
    raise ImportError("the fake player needs pillow to encode frames") from error


IMAGE_SIZE = 84
GOAL = (-10.0, 60.0)
START = (-10.0, 40.0)
GOAL_REACH_DISTANCE = 2.5
FORWARD_STEP = 1.25
BRAIN_NAME = "TVBrain"
RESET_PARAMETERS = {"startLoc": 0.0, "render": 0.0, "door": 1.0, "tv": 1.0}
BUFFER_SIZE = 120000


class Channel:
    """Buffered byte-stream view of the connection to the client.

    The upstream client is not message-framed on the wire, and TCP may deliver
    its ``.`` acknowledgment, a command, and a small JSON payload in a single
    segment; this class reassembles them.
    """

    def __init__(self, connection: socket.socket) -> None:
        """Wrap a connected socket.

        Input: Connected socket.
        Output: A channel with an empty buffer.
        Mathematical meaning: None; protocol transport.
        """
        self.connection = connection
        self.buffer = b""

    def _fill(self) -> None:
        """Read one more chunk from the socket.

        Input: This channel.
        Output: No value; the buffer grows or ``EOFError`` is raised.
        Mathematical meaning: None; protocol transport.
        """
        chunk = self.connection.recv(BUFFER_SIZE)
        if not chunk:
            raise EOFError("the client closed the connection")
        self.buffer += chunk

    def read_acknowledgment(self, message: str = "received ack") -> bytes:
        """Consume one ``RECEIVED``-style handshake.

        Input: Optional trace label.
        Output: The consumed token.
        Mathematical meaning: None; protocol transport.
        """
        return self.read_token((b"RECEIVED", b".", b"ACK"), message)

    def read_token(self, tokens: Iterable[bytes], label: str = "received token") -> bytes:
        """Consume the next occurrence of any expected token.

        Input: Accepted tokens in priority order and an optional trace label.
        Output: The token that was consumed.
        Mathematical meaning: None; protocol transport.
        """
        while True:
            for token in tokens:
                if self.buffer.startswith(token):
                    self.buffer = self.buffer[len(token) :]
                    _log(f"{label}: {token!r}")
                    return token
            self._fill()

    def read_command(self) -> bytes:
        """Consume the next command sent by the client.

        Input: This channel.
        Output: One of ``b"RESET"``, ``b"STEP"``, or ``b"EXIT"``.
        Mathematical meaning: None; protocol transport.
        """
        return self.read_token((b"RESET", b"STEP", b"EXIT"), "received command")

    def read_json(self, label: str = "received json") -> Any:
        """Consume the next JSON value, tolerating segmentation.

        Input: Optional trace label.
        Output: The decoded JSON value.
        Mathematical meaning: None; protocol transport.
        """
        decoder = json.JSONDecoder()
        while True:
            try:
                text = self.buffer.decode("utf-8")
            except UnicodeDecodeError:
                self._fill()
                continue
            try:
                value, index = decoder.raw_decode(text)
            except ValueError:
                self._fill()
                continue
            consumed = len(text[:index].encode("utf-8"))
            self.buffer = self.buffer[consumed:]
            _log(f"{label}: {text[:index][:160]}")
            return value


def _first_action(payload: Any, brain_name: str) -> int:
    """Extract the discrete action for one brain from the client's message.

    Input: The ``action`` field of the client's step message, which is either a
        mapping keyed by brain name or a plain list, and the brain name.
    Output: The integer action for that brain.
    Mathematical meaning: Recovers ``a_t`` from the transport encoding.
    """
    if isinstance(payload, dict):
        payload = payload.get(brain_name, payload)
    array = np.asarray(payload if payload is not None else [0]).reshape(-1)
    return int(array[0]) if array.size else 0


def _log(message: str) -> None:
    """Append a protocol trace when ``NOISY_TV_FAKE_PLAYER_LOG`` is set.

    Input: Human-readable message.
    Output: No value; the message is written to the log file when configured.
    Mathematical meaning: None; diagnostics for the test harness.
    """
    path = os.environ.get("NOISY_TV_FAKE_PLAYER_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        # Tracing is diagnostic only: the harness may have removed its
        # temporary directory while this process was shutting down, and that
        # must not turn into an error in the player.
        pass


class FakeTVMaze:
    """The player-side episode model and protocol state machine."""

    def __init__(self, connection: socket.socket) -> None:
        """Store the accepted connection and initialize the episode state.

        Input: Connected socket accepted from the client.
        Output: A player ready to answer commands.
        Mathematical meaning: Defines the MDP the fake player samples from.
        """
        self.channel = Channel(connection)
        self.parameters = dict(RESET_PARAMETERS)
        self.position = START
        self.channel_index = 0
        self.steps = 0
        self.done = False
        self.rng = np.random.default_rng(0)
        self.screen = self.draw_channel()

    # ----------------------------------------------------------------- state
    def draw_channel(self) -> np.ndarray:
        """Draw the pattern currently shown on the television.

        Input: This player's random generator.
        Output: ``uint8`` array shaped ``(18, 18, 3)``.
        Mathematical meaning: A sample of the screen content. The caller
            decides when a new sample is drawn, which is what separates the
            noisy television (a fresh sample every step) from the deterministic
            one (a fresh sample only when the channel changes).
        """
        return self.rng.integers(0, 256, size=18 * 18 * 3, dtype=np.uint8).reshape(18, 18, 3)

    def frame(self) -> np.ndarray:
        """Render the corridor from the agent's position.

        Input: This player's position and the television's current screen.
        Output: ``uint8`` array shaped ``(84, 84, 3)``.
        Mathematical meaning: The observation the client transmits. It is a
            deterministic function of the state, so identical states produce
            identical frames - the property the upstream camera has too.
        """
        frame = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 40, dtype=np.uint8)
        frame[:, IMAGE_SIZE // 3 : 2 * IMAGE_SIZE // 3] = 150
        distance = abs(self.position[1] - GOAL[1])
        horizon = int(np.clip((1.0 - distance / 25.0) * IMAGE_SIZE, 2, IMAGE_SIZE - 1))
        frame[IMAGE_SIZE // 2 - horizon // 2 : IMAGE_SIZE // 2 + horizon // 2, :, :] = 60
        frame[62:80, 62:80] = self.screen
        return frame

    def send_state(self) -> None:
        """Send the full state of one step, honouring the client handshakes.

        Input: This player's current state.
        Output: No value; the state dictionary, one frame, and the global-done
            flag are written, each awaited by the client.
        Mathematical meaning: Transmits one sample of the environment state.
        """
        distance = float(np.hypot(self.position[0] - GOAL[0], self.position[1] - GOAL[1]))
        reward = 1.0 if distance < GOAL_REACH_DISTANCE else 0.0
        self.done = reward > 0.0
        message = json.dumps(
            {
                "brain_name": BRAIN_NAME,
                "agents": [0],
                "states": [float(self.position[0]), float(self.position[1])],
                "memories": [],
                "rewards": [reward],
                "dones": [bool(self.done)],
            }
        ).encode("utf-8")
        self.channel.connection.send(message)
        self.channel.read_acknowledgment("state json ack")
        buffer = io.BytesIO()
        Image.fromarray(self.frame()).save(buffer, format="PNG")
        self.channel.connection.send(buffer.getvalue())
        self.channel.read_acknowledgment("frame ack")
        self.channel.connection.send(b"True" if self.done else b"False")

    # ------------------------------------------------------------- commands
    def reset(self) -> None:
        """Start a new episode using the client's reset parameters.

        Input: None; the parameters arrive over the connection.
        Output: No value; the initial state is sent.
        Mathematical meaning: Samples ``s_0`` from the initial-state
            distribution selected by ``startLoc``/``door``/``tv``.
        """
        self.channel.connection.send(b"ACK")
        request = self.channel.read_json("reset parameters")
        for key, value in (request.get("parameters") or {}).items():
            if key in self.parameters:
                self.parameters[key] = float(value)
        self.position = (START[0], START[1] + 5.0 * self.parameters.get("startLoc", 0.0))
        self.channel_index = 0
        self.steps = 0
        self.done = False
        self.screen = self.draw_channel()
        self.send_state()

    def step(self) -> None:
        """Advance the episode by one action.

        Input: None; the action arrives over the connection.
        Output: No value; the next state is sent.
        Mathematical meaning: Applies the upstream action semantics: ``1``
            drives forward, ``5`` changes the channel, and a noisy television
            redraws every step.
        """
        self.channel.connection.send(b"ACK")
        message = self.channel.read_json("action")
        # The client keys every field by brain name for a single-brain academy:
        # {"action": {"TVBrain": [1.0]}, "memory": {...}, "value": {...}}.
        action = _first_action(message.get("action"), BRAIN_NAME)
        loud = float(self.parameters.get("tv", 1.0)) == 1.0
        if action == 1 and not self.done:
            self.position = (self.position[0], self.position[1] + FORWARD_STEP)
        elif action == 5:
            # Action 5 is the original build's channel switch.
            self.channel_index += 1
            self.screen = self.draw_channel()
        if loud and not self.done:
            # The noisy television is unpredictable: its screen is resampled on
            # every step regardless of what the agent does.
            self.screen = self.draw_channel()
        self.steps += 1
        self.send_state()

    def serve(self) -> None:
        """Run the protocol loop until the client says ``EXIT``.

        Input: This player with an accepted connection.
        Output: No value; the process ends when the client closes.
        Mathematical meaning: None; protocol transport.
        """
        self.channel.connection.send(
            json.dumps(
                {
                    "AcademyName": "TemplateAcademy",
                    "brainNames": [BRAIN_NAME],
                    "resetParameters": self.parameters,
                    "brainParameters": [
                        {
                            "stateSize": 2,
                            "actionSize": 6,
                            "memorySize": 0,
                            "cameraResolutions": [
                                {"width": IMAGE_SIZE, "height": IMAGE_SIZE, "blackAndWhite": 0}
                            ],
                            "actionDescriptions": [],
                            "actionSpaceType": 0,
                            "stateSpaceType": 1,
                        }
                    ],
                }
            ).encode("utf-8")
        )
        self.channel.read_acknowledgment("handshake")
        while True:
            command = self.channel.read_command()
            if command == b"RESET":
                self.reset()
            elif command == b"STEP":
                self.step()
            else:
                _log("EXIT command received")
                return


def main(argv: Optional[list] = None) -> int:
    """Connect to the client's socket and serve the protocol.

    Input: Command-line arguments, containing ``--port``.
    Output: Process exit status.
    Mathematical meaning: None; the fake player only transports state.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args(argv)

    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.connect(("localhost", arguments.port))
    _log(f"player started, pid={os.getpid()}, port={arguments.port}")
    try:
        FakeTVMaze(connection).serve()
    except (ConnectionResetError, BrokenPipeError, EOFError):
        _log("connection closed by the client")
        return 0
    finally:
        connection.close()
        _log("player terminated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
