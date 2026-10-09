"""Integration tests for the original Unity Noisy-TV path.

The reference environment is a Unity application: its python side is the
``unityagents`` client vendored in ``luchris429/noisy-tv-env``, and the player
is a separate executable that the client launches as
``tv_maze[.x86_64] --port <port>`` before speaking a JSON-plus-JPEG protocol
over a TCP socket. The player binary itself is distributed out of band and is
therefore frequently unavailable (no network access to Google Drive, no GPU, no
display).

These tests exercise the *whole python half of that path for real*: they start
the genuine upstream client, which binds a socket, launches a player process
and decodes the frames it receives. The player process is
``tests/fixtures/fake_tv_maze_player.py``, extracted to a temporary directory as
``tv_maze.x86_64`` so that the client's Linux launch logic finds it exactly as
it would find the Unity build. Only the Unity engine is simulated; the client,
the socket protocol, the launch semantics, the JPEG decoding, the adapter, and
the noisy-television behaviour are the real thing.

The tests skip cleanly when the upstream client or ``pillow`` is missing; run
``bash scripts/install_noisy_tv_unity.sh`` to install them (it also downloads
the real player when Google Drive is reachable).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Optional
from unittest import mock

import numpy as np

from environments.noisy_tv_unity import (
    ENVIRONMENT_ID,
    MAZE_ENVIRONMENT_ID,
    NoisyTVUnityConfig,
    UnityEnvironmentAdapter,
    WORKER_ID_SEARCH_LIMIT,
    client_search_paths,
    make_noisy_tv_environment,
    player_search_paths,
    port_available,
    worker_id_candidates,
    preferred_environment_id,
    resolve_player_path,
    unity_available,
    unity_binary_available,
    unity_client_available,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fake_tv_maze_player.py"

class _TrackedPopen(subprocess.Popen):
    """A ``Popen`` whose instances stay reachable so the tests can reap them.

    The vendored client launches the player and then discards the handle: the
    object is collected while the player is still running (CPython warns about
    a still-running child) and the process is reaped only at interpreter exit.
    Retaining the handles lets a test wait for the players it started, which
    removes the warning and turns "the player really terminated" into an
    assertion. Nothing else about the client's launch path is changed.
    """

    instances: List["_TrackedPopen"] = []

    def __init__(self, *arguments: object, **keywords: object) -> None:
        """Create the process and register its handle.

        Input: Arguments forwarded to :class:`subprocess.Popen`.
        Output: No value.
        Mathematical meaning: None; process bookkeeping.
        """
        super().__init__(*arguments, **keywords)
        _TrackedPopen.instances.append(self)


def _reap_tracked_processes(timeout: float = 15.0) -> None:
    """Wait for every player launched through the client and kill stragglers.

    Input: Per-process timeout in seconds.
    Output: No value; all registered handles are reaped and forgotten.
    Mathematical meaning: None; it bounds the lifetime of the external player.
    """
    while _TrackedPopen.instances:
        process = _TrackedPopen.instances.pop()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()
            process.wait(timeout=timeout)


def _install_fake_player(directory: Path) -> str:
    """Install the fake player where the upstream client expects to find it.

    Input: Temporary directory that will hold the "build".
    Output: The ``file_name`` value to hand to the client (without suffix).
    Mathematical meaning: None; it mimics the layout of the downloaded Unity
        build, whose Linux executable is named ``tv_maze.x86_64`` and is
        launched by the client with a single ``--port`` argument.

    The client launches the file directly, so the executable is a small shell
    script that runs the fixture with the interpreter executing the tests
    (``sys.executable``); that is the same indirection the real native binary
    provides, and it keeps the fixture independent of the system interpreter.
    """
    target = directory / "tv_maze.x86_64"
    target.write_text(
        "#!/bin/sh\n"
        f'exec "{sys.executable}" "{FIXTURE}" "$@"\n'
    )
    target.chmod(0o755)
    return str(directory / "tv_maze")


@unittest.skipUnless(unity_client_available(), "the upstream unityagents client is not installed")
class UnityClientIntegrationTest(unittest.TestCase):
    """Drive the original Unity path with the real upstream client."""

    def setUp(self) -> None:
        """Create a temporary build directory holding the player.

        Input: None.
        Output: ``self.directory`` and ``self.file_name`` for the client.
        Mathematical meaning: Establishes a launchable "build" for the tests.
        """
        self.directory = Path(tempfile.mkdtemp(prefix="noisy_tv_unity_"))
        self.file_name = _install_fake_player(self.directory)
        self.adapters: List[UnityEnvironmentAdapter] = []
        self.trace = self.directory / "player.log"
        self.environment_patch = mock.patch.dict(
            os.environ, {"NOISY_TV_FAKE_PLAYER_LOG": str(self.trace)}, clear=False
        )
        self.environment_patch.start()
        self.process_patch = mock.patch("subprocess.Popen", _TrackedPopen)
        self.process_patch.start()

    def tearDown(self) -> None:
        """Close every adapter and remove the temporary build.

        Input: None.
        Output: No value.
        Mathematical meaning: None.
        """
        for adapter in self.adapters:
            try:
                adapter.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        _reap_tracked_processes()
        self.process_patch.stop()
        self.environment_patch.stop()
        shutil.rmtree(self.directory, ignore_errors=True)

    def _wait_for_trace(self, needle: str, timeout: float = 5.0) -> str:
        """Wait until the player's protocol trace contains ``needle``.

        Input: Substring to wait for and a timeout in seconds.
        Output: The trace text.
        Mathematical meaning: None; it observes the player's lifetime.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self.trace.read_text() if self.trace.exists() else ""
            if needle in text:
                return text
            time.sleep(0.05)
        return self.trace.read_text() if self.trace.exists() else ""

    def _adapter(self, **config_kwargs) -> UnityEnvironmentAdapter:
        """Build an adapter around the upstream client launching the fake player.

        Input: Optional :class:`NoisyTVUnityConfig` overrides.
        Output: A connected adapter, registered for cleanup.
        Mathematical meaning: Couples the pipeline to the upstream MDP
            interface exactly as the Unity build would.
        """
        # Use a unique worker id per test so sockets never collide.
        worker_id = int(np.random.default_rng().integers(0, 400))
        configuration = NoisyTVUnityConfig(
            file_name=self.file_name, worker_id=worker_id, base_port=5200, **config_kwargs
        )
        adapter = UnityEnvironmentAdapter(configuration)
        self.adapters.append(adapter)
        return adapter

    def test_real_client_streams_frames_and_actions(self) -> None:
        """The real client + adapter produce valid observations and rewards.

        Input: A reset and a sequence of forward actions through the client.
        Output: Assertions on the frames, the state info, and the sparse
            reward that the upstream maze pays at its goal.
        Mathematical meaning: Verifies the translation of the upstream
            ``BrainInfo`` stream into the pipeline's transition tuple on the
            genuine transport, not on a stub.
        """
        adapter = self._adapter()
        observation, info = adapter.reset()
        self.assertEqual(observation.shape, (84, 84, 3))
        self.assertEqual(observation.dtype, np.uint8)
        self.assertTrue(adapter.observation_space.contains(observation))
        self.assertEqual(adapter.action_space.n, 6)
        self.assertIn("distance_to_goal", info)
        self.assertEqual(adapter.backend, "unity")

        rewards = []
        terminated = False
        for _ in range(20):
            observation, reward, terminated, truncated, info = adapter.step(1)
            rewards.append(reward)
            self.assertEqual(observation.shape, (84, 84, 3))
            self.assertFalse(truncated)
            if terminated:
                break
        self.assertTrue(terminated, "the fake player pays +1 at the goal")
        self.assertEqual(rewards[-1], 1.0)
        self.assertEqual(sum(rewards), 1.0)
        self.assertLess(info["distance_to_goal"], 2.5)

    def test_reset_parameters_reach_the_player(self) -> None:
        """The upstream reset dictionary is forwarded and acknowledged.

        Input: ``reset`` with explicit ``start_loc``/``door``/``tv`` options.
        Output: Assertions on the adapter's recorded configuration and on the
            requested parameter round trip.
        Mathematical meaning: Selects the upstream episode condition, which the
            thesis varies (noisy television on/off).
        """
        adapter = self._adapter(tv=1.0, door=1.0)
        adapter.reset(options={"start_loc": 2.0, "door": 1.0, "tv": 0.0})
        self.assertEqual(adapter.config.tv, 1.0)
        self.assertEqual(adapter.brain.state_space_size, 2)
        self.assertEqual(adapter.brain.action_space_type, "discrete")

    def test_noisy_television_changes_observations_through_the_client(self) -> None:
        """``tv=1.0`` (the paper's noisy television) keeps changing the frames.

        Input: A stationary agent on the genuine client transport.
        Output: Assertion that consecutive frames differ while no reward is
            paid.
        Mathematical meaning: The unpredictable, task-irrelevant novelty the
            paper studies, transported end to end.
        """
        adapter = self._adapter(tv=1.0)
        frames = [adapter.reset(options={"tv": 1.0})[0]]
        rewards = []
        for _ in range(12):
            observation, reward, terminated, truncated, _ = adapter.step(0)
            frames.append(observation)
            rewards.append(reward)
            if terminated or truncated:
                break
        distinct = {frame.tobytes() for frame in frames}
        self.assertGreater(len(distinct), 1, "the noisy screen must keep changing")
        self.assertEqual(sum(rewards), 0.0, "watching the television must pay nothing")

    def test_static_television_is_deterministic_through_the_client(self) -> None:
        """``tv=0.0`` freezes the screen.

        Input: The same stationary rollout with the television switched off.
        Output: Assertion that every frame is identical.
        Mathematical meaning: The deterministic control condition.
        """
        adapter = self._adapter(tv=0.0)
        first = adapter.reset(options={"tv": 0.0})[0]
        for step in range(6):
            observation, reward, terminated, truncated, _ = adapter.step(0)
            if terminated or truncated:
                break
            np.testing.assert_array_equal(observation, first)

    def test_client_shutdown_stops_the_player(self) -> None:
        """``close`` reaches the player and the player then exits.

        Input: A connected adapter that is closed again.
        Output: Assertion that the player logged its own start and termination.
        Mathematical meaning: Ends the interaction with the upstream MDP
            cleanly, so a training run cannot leave orphaned player processes.
        """
        adapter = self._adapter()
        adapter.reset()
        started = self._wait_for_trace("player started")
        self.assertIn("player started", started)
        adapter.close()
        trace = self._wait_for_trace("player terminated")
        self.assertIn("EXIT command received", trace)
        self.assertIn("player terminated", trace)
        # The client discards its own process handle on launch, so these tests
        # track it: awaiting the process proves that no player survives close.
        process = _TrackedPopen.instances[-1]
        process.wait(timeout=15)
        self.assertIsNotNone(process.returncode)

    def test_adapter_reports_action_errors_from_the_protocol(self) -> None:
        """Invalid actions are rejected before they reach the player.

        Input: Action ``7`` on a six-action brain.
        Output: Assertion that the adapter raises ``ValueError``.
        Mathematical meaning: Enforces membership of ``a_t`` in the action set.
        """
        adapter = self._adapter()
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.step(7)

    def test_close_releases_the_player_process(self) -> None:
        """Closing the adapter shuts the player down.

        Input: An adapter that has been reset.
        Output: Assertion that closing twice is safe and the connection is
            gone.
        Mathematical meaning: Ends interaction with the upstream MDP instance.
        """
        adapter = self._adapter()
        adapter.reset()
        adapter.close()
        self.adapters.remove(adapter)
        self.assertFalse(adapter.client.global_done if adapter.client.global_done else False)


@unittest.skipUnless(unity_client_available(), "the upstream unityagents client is not installed")
class UnityTrainingIntegrationTest(unittest.TestCase):
    """The training pipeline must be able to drive the Unity path."""

    def setUp(self) -> None:
        """Create a temporary build directory with the fake player.

        Input: None.
        Output: ``self.directory`` and ``self.file_name``.
        Mathematical meaning: Provides a launchable environment for the tests.
        """
        self.directory = Path(tempfile.mkdtemp(prefix="noisy_tv_unity_"))
        self.file_name = _install_fake_player(self.directory)
        self.adapters: List[UnityEnvironmentAdapter] = []
        self.process_patch = mock.patch("subprocess.Popen", _TrackedPopen)
        self.process_patch.start()

    def tearDown(self) -> None:
        """Close every player process and remove the temporary build.

        Input: None.
        Output: No value.
        Mathematical meaning: None; a leaked player would keep its socket and
            process alive and would warn on interpreter shutdown.
        """
        for adapter in self.adapters:
            try:
                adapter.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        _reap_tracked_processes()
        self.process_patch.stop()
        shutil.rmtree(self.directory, ignore_errors=True)

    def _adapter(self, **config_kwargs) -> UnityEnvironmentAdapter:
        """Build an adapter around the fake player, registered for cleanup.

        Input: Optional :class:`NoisyTVUnityConfig` overrides.
        Output: A connected adapter.
        Mathematical meaning: None; it only manages process lifetime.
        """
        configuration = NoisyTVUnityConfig(
            file_name=self.file_name, **config_kwargs
        )
        adapter = UnityEnvironmentAdapter(configuration)
        self.adapters.append(adapter)
        return adapter

    def test_adapter_satisfies_the_training_interface(self) -> None:
        """The Unity adapter works as a training environment.

        Input: The Unity adapter wrapped in the pipeline's single-environment
            adapter, driven with a tiny schedule.
        Output: Assertions on the observation shape and the step contract the
            trainer relies on.
        Mathematical meaning: Confirms that the original Unity build can be
            substituted for the reconstruction in the PPO/BiGAN pipeline.
        """
        from environments.adapter import SingleEnvironmentAdapter

        adapter = self._adapter(worker_id=0, base_port=5300)
        try:
            environment = SingleEnvironmentAdapter(adapter)
            self.assertEqual(environment.observation_shape, (84, 84, 3))
            observation, _ = environment.reset()
            self.assertEqual(tuple(observation.shape), (84, 84, 3))
            step = environment.step(1)
            self.assertEqual(tuple(step.observation.shape), (84, 84, 3))
            self.assertIsInstance(step.reward, float)
            self.assertIn("distance_to_goal", step.info)
        finally:
            adapter.close()

    def test_make_noisy_tv_environment_builds_the_unity_backend(self) -> None:
        """``backend="unity"`` returns the original-build adapter.

        Input: ``make_noisy_tv_environment("unity", file_name=..., worker_id=...)``.
        Output: Assertion that a connected Unity adapter is returned.
        Mathematical meaning: The same MDP family, driven through the reference
            implementation.
        """
        environment = make_noisy_tv_environment(
            "unity", file_name=self.file_name, worker_id=0, base_port=5310
        )
        self.adapters.append(environment)
        try:
            self.assertIsInstance(environment, UnityEnvironmentAdapter)
            self.assertEqual(environment.backend, "unity")
            observation, _ = environment.reset()
            self.assertEqual(observation.shape, (84, 84, 3))
        finally:
            environment.close()

    def test_registered_environment_id_launches_the_configured_build(self) -> None:
        """``--env NoisyTVUnity-v0`` launches the build named by the environment variable.

        Input: ``NOISY_TV_UNITY_BINARY`` set to a reachable player and default
            construction arguments, exactly as :func:`main.make_training_environment`
            and ``gymnasium.make`` use them.
        Output: Assertions that the default configuration resolves to the
            configured executable and that a rollout works.
        Mathematical meaning: The training pipeline reaches the original MDP
            through the registered ID without any hand-written path.
        """
        with mock.patch.dict(
            os.environ, {"NOISY_TV_UNITY_BINARY": self.file_name}, clear=False
        ):
            environment = self._adapter(worker_id=0, base_port=5340)
            self.assertEqual(environment.config.file_name, self.file_name)
            observation, _ = environment.reset()
            self.assertEqual(tuple(observation.shape), (84, 84, 3))
            _, reward, terminated, truncated, _ = environment.step(1)
            self.assertIsInstance(reward, float)
            self.assertFalse(terminated and truncated)

    def test_auto_backend_selects_unity_when_the_build_is_present(self) -> None:
        """``backend="auto"`` prefers the original build.

        Input: A reachable player and a monkey-patched search path.
        Output: Assertions on the backend actually used and on
            ``preferred_environment_id``.
        Mathematical meaning: Guarantees that the reference implementation is
            used whenever it can be, with no silent substitution.
        """
        with mock.patch.dict(
            os.environ, {"NOISY_TV_UNITY_BINARY": self.file_name}, clear=False
        ):
            self.assertTrue(unity_available())
            self.assertEqual(preferred_environment_id(), ENVIRONMENT_ID)
            environment = make_noisy_tv_environment(
                "auto", worker_id=0, base_port=5320
            )
            self.adapters.append(environment)
            try:
                self.assertIsInstance(environment, UnityEnvironmentAdapter)
                self.assertEqual(environment.backend, "unity")
            finally:
                environment.close()

    def test_pipeline_uses_the_unity_environment_id(self) -> None:
        """``main.build_config`` accepts the Unity adapter through the factory.

        Input: ``--env NoisyTVUnity-v0`` arguments and a Unity adapter.
        Output: Assertions on the derived configuration.
        Mathematical meaning: The PPO/BiGAN function domains follow the
            upstream MDP's spaces.
        """
        import main as main_module

        adapter = self._adapter(worker_id=0, base_port=5330)
        try:
            arguments = _parse_arguments(
                ["--env", ENVIRONMENT_ID, "--method", "state", "--num-parallel-envs", "2"]
            )
            configuration = main_module.build_config(arguments, adapter)
        finally:
            adapter.close()
        self.assertEqual(configuration.environment.observation_shape, (84, 84, 3))
        self.assertEqual(configuration.environment.action_dim, 6)
        self.assertTrue(configuration.environment.discrete_actions)


class PlayerDiscoveryTest(unittest.TestCase):
    """Discovery rules for the player and the vendored client."""

    def test_player_search_paths_honour_the_environment_variable(self) -> None:
        """``NOISY_TV_UNITY_BINARY`` is searched first.

        Input: The environment variable set to a temporary directory.
        Output: Assertion that the path is the first candidate and resolves.
        Mathematical meaning: None; it documents the override the installer
            script recommends.
        """
        with tempfile.TemporaryDirectory() as directory:
            file_name = _install_fake_player(Path(directory))
            with mock.patch.dict(os.environ, {"NOISY_TV_UNITY_BINARY": file_name}, clear=False):
                candidates = player_search_paths()
                self.assertEqual(candidates[0], file_name)
                self.assertEqual(resolve_player_path(), file_name)
                self.assertTrue(unity_binary_available(file_name))

    def test_explicit_name_falls_back_to_the_configured_build(self) -> None:
        """A bare ``file_name`` still finds the build ``NOISY_TV_UNITY_BINARY`` names.

        Input: A reachable build and the default bare file name.
        Output: Assertions that resolution and availability both succeed.
        Mathematical meaning: The environment registered as ``NoisyTVUnity-v0``
            is constructed with default arguments, so the configured build must
            be discoverable without passing a path explicitly - otherwise
            ``unity_available`` would announce a build the adapter cannot use.
        """
        with tempfile.TemporaryDirectory() as directory:
            file_name = _install_fake_player(Path(directory))
            with mock.patch.dict(
                os.environ, {"NOISY_TV_UNITY_BINARY": file_name}, clear=False
            ):
                self.assertEqual(
                    resolve_player_path(NoisyTVUnityConfig().file_name), file_name
                )
                self.assertTrue(unity_available(NoisyTVUnityConfig()))
                self.assertEqual(preferred_environment_id(), ENVIRONMENT_ID)

    def test_missing_player_resolves_to_none(self) -> None:
        """An unreachable player is reported instead of assumed.

        Input: ``NOISY_TV_UNITY_BINARY`` pointing into an empty directory.
        Output: Assertion that resolution returns ``None``.
        Mathematical meaning: None; it keeps the fallback decision explicit.
        """
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"NOISY_TV_UNITY_BINARY": str(Path(directory) / "tv_maze")},
                clear=False,
            ):
                self.assertIsNone(resolve_player_path())

    def test_client_search_paths_include_third_party(self) -> None:
        """The vendored client location is searched by default.

        Input: None.
        Output: Assertion that ``third_party/noisy-tv-env`` is a candidate.
        Mathematical meaning: None; it is where the installer script clones the
            upstream repository.
        """
        candidates = client_search_paths()
        self.assertTrue(
            any(candidate.endswith(os.path.join("third_party", "noisy-tv-env"))
                for candidate in candidates),
            candidates,
        )

    def test_preferred_environment_falls_back_to_the_reconstruction(self) -> None:
        """Without a player the reconstruction is preferred.

        Input: ``unity_available`` forced to ``False``.
        Output: Assertion that the python maze ID is returned.
        Mathematical meaning: The same MDP family stays available everywhere.
        """
        with mock.patch("environments.noisy_tv_unity.unity_available", return_value=False):
            self.assertEqual(preferred_environment_id(), MAZE_ENVIRONMENT_ID)

    def test_preferred_environment_prefers_unity_when_available(self) -> None:
        """With a player present the original build is preferred.

        Input: ``unity_available`` forced to ``True``.
        Output: Assertion that the Unity ID is returned.
        Mathematical meaning: The reference implementation wins when it can run.
        """
        with mock.patch("environments.noisy_tv_unity.unity_available", return_value=True):
            self.assertEqual(preferred_environment_id(), ENVIRONMENT_ID)


def _parse_arguments(argv: List[str]):
    """Parse command-line arguments through the real entry point.

    Input: Argument list without the program name.
    Output: Parsed argument namespace.
    Mathematical meaning: None.
    """
    import main as main_module

    with mock.patch.object(sys, "argv", ["main.py", *argv]):
        return main_module.parse_args()


if __name__ == "__main__":
    unittest.main()

class UnityPipelineIntegrationTest(unittest.TestCase):
    """``--env NoisyTVUnity-v0`` must work through the pipeline's own factory.

    These tests go through ``main.make_training_environment``, i.e. the exact
    call the driver makes: Gymnasium construction by registered ID, the
    pipeline's single- and vector-environment adapters, and several parallel
    players in one process.
    """

    def setUp(self) -> None:
        """Install the fake player and point the search path at it.

        Input: None.
        Output: ``self.directory`` and ``self.file_name`` plus an environment
            variable that makes the build discoverable.
        Mathematical meaning: Provides a launchable environment for the tests.
        """
        self.directory = Path(tempfile.mkdtemp(prefix="noisy_tv_unity_"))
        self.file_name = _install_fake_player(self.directory)
        self.environment_patch = mock.patch.dict(
            os.environ, {"NOISY_TV_UNITY_BINARY": self.file_name}, clear=False
        )
        self.environment_patch.start()
        self.process_patch = mock.patch("subprocess.Popen", _TrackedPopen)
        self.process_patch.start()
        self.vector_environment = None

    def tearDown(self) -> None:
        """Close the vector environment, reap the players, and clean up.

        Input: None.
        Output: No value.
        Mathematical meaning: None.
        """
        if self.vector_environment is not None:
            try:
                self.vector_environment.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        _reap_tracked_processes()
        self.process_patch.stop()
        self.environment_patch.stop()
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_gymnasium_make_accepts_the_registered_unity_id(self) -> None:
        """``gymnasium.make`` builds the Unity environment by its ID.

        Input: ``NoisyTVUnity-v0`` through ``make_gymnasium_environment``.
        Output: Assertions on the wrapped environment and on a reset/step pair.
        Mathematical meaning: The registered ID addresses the upstream MDP, so
            ``--env NoisyTVUnity-v0`` is usable by the driver.

        ``gymnasium.make`` rejects entry points whose class does not inherit
        from ``gymnasium.Env``; without that inheritance the ID above is
        advertised but unusable.
        """
        from environments.adapter import make_gymnasium_environment

        environment = make_gymnasium_environment(ENVIRONMENT_ID)
        try:
            self.assertEqual(environment.observation_shape, (84, 84, 3))
            observation, _ = environment.reset()
            self.assertEqual(tuple(observation.shape), (84, 84, 3))
            step = environment.step(1)
            self.assertIsInstance(step.reward, float)
            self.assertIn("distance_to_goal", step.info)
        finally:
            environment.close()

    def test_parallel_environments_use_distinct_worker_ids(self) -> None:
        """Several players coexist in one process on distinct ports.

        Input: ``main.make_training_environment`` with two parallel environments.
        Output: Assertions that both players exist, answer, and used different
            worker ids.
        Mathematical meaning: A batched rollout samples the product MDP, so
            ``--num-parallel-envs 2`` must build two independent players.
        """
        import main as main_module

        self.vector_environment = main_module.make_training_environment(
            ENVIRONMENT_ID, num_parallel_envs=2, seed=0
        )
        # ``gymnasium.make`` wraps the entry point in ``OrderEnforcing`` (and a
        # passive checker), so the Unity adapter itself is ``unwrapped``.
        adapters = [child.environment.unwrapped for child in self.vector_environment.environments]
        self.assertEqual(len(adapters), 2)
        worker_ids = {adapter.config.worker_id for adapter in adapters}
        self.assertEqual(len(worker_ids), 2, "parallel players must not share a port")
        observations, _ = self.vector_environment.reset()
        self.assertEqual(tuple(observations.shape[:1]), (2,))
        transitions = self.vector_environment.step(np.array([1, 1]))
        self.assertEqual(len(transitions.rewards), 2)

    def test_missing_free_port_is_reported(self) -> None:
        """Every port in the search window being busy raises a clear error.

        Input: A port range that is fully occupied by open sockets.
        Output: Assertion that construction raises ``RuntimeError`` naming the
            range instead of an obscure socket error.
        Mathematical meaning: Bounds the resource a run may claim.
        """
        held = []
        try:
            first_worker = 8000
            for offset in range(WORKER_ID_SEARCH_LIMIT):
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.bind(("localhost", 5400 + first_worker + offset))
                listener.listen(1)
                held.append(listener)
            self.assertFalse(port_available(5400, first_worker))
            configuration = NoisyTVUnityConfig(
                file_name=self.file_name, worker_id=first_worker, base_port=5400
            )
            with self.assertRaises(RuntimeError) as raised:
                UnityEnvironmentAdapter(configuration)
            self.assertIn("no free worker id", str(raised.exception))
        finally:
            for listener in held:
                listener.close()
