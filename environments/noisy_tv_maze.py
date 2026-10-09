"""NoisyTVMaze: the Noisy-TV maze of "Large-Scale Study of Curiosity-Driven Learning".

Upstream environment
--------------------
The reference implementation is the Unity project in
`luchris429/noisy-tv-env <https://github.com/luchris429/noisy-tv-env>`_, which
describes itself as "The Noisy TV Environment from *Large-Scale Study of
Curiosity-Driven Learning* (ICLR 2019)" (Burda, Edwards, Pathak, Storkey,
Darrell and Efros). An agent navigates a maze of rooms and corridors that
contains a television. The television keeps displaying images drawn at random,
so its content is *unpredictable and irrelevant* to the task; the paper uses it
to exhibit the failure mode in which prediction-error curiosity is captured by
the stochastic distractor instead of by the reward.

This module reproduces that environment as a Gymnasium task without a Unity
dependency, so the thesis pipeline can be developed, smoke-tested and
regression-tested on a machine with neither a GPU nor a display. The geometry
below was not invented: :data:`WALL_BOXES` is the maze's 138 wall cubes read
out of the upstream scene ``Assets/tv_maze.unity`` (each cube's world
position, size and orientation, with rotated cubes converted to axis-aligned
footprints), and :data:`START_POSES`, the television plane, the goal sphere,
the sliding door and the button logic come from
``Assets/Template/Scripts/TVAgent.cs``, ``TemplateAcademy.cs``, and
``Assets/SlidingDoor.cs``.

Upstream interface that is reproduced
-------------------------------------
==============================  ===========================================
upstream declaration            reconstruction
==============================  ===========================================
``cameraResolutions: 84x84``    ``Box(0, 255, (84, 84, 3), uint8)``
``blackAndWhite: 0``            RGB, three channels
``stateSize: 2``                ``(x, z)`` pose reported through ``info``
``actionSize: 6``, discrete     ``Discrete(6)`` with upstream meanings
``startLoc``                    ``start_loc`` (``0`` = random start pose)
``door``                        ``door``: ``0`` opened and randomized,
                                ``1`` closed and deterministic, ``2`` closed
                                and randomized
``tv``                          ``tv``: ``0`` static screen, ``1`` screen that
                                redraws itself (noisy), ``2`` static screen
                                the agent may change
reward ``1.0`` within ``2.5``   identical, with episode termination
of the goal
==============================  ===========================================

Actions (``TVAgent.AgentStep``): ``0`` do nothing, ``1`` drive forward at
``5.0`` units/s, ``2`` rotate left, ``3`` rotate right, ``4`` press the sliding
door, ``5`` change the television channel. The upstream agent counts door and
channel presses with one shared counter and only actuates every tenth press
(``count = (count + 1) % 10``, starting at ``3``), and the television only
answers within ``18.0`` units; both quirks are reproduced, because they are part
of what the agent has to discover.

Maze topology (verified on the upstream geometry)
-------------------------------------------------
The maze is sealed: a flood fill from any start pose reaches every other start
pose, the goal sphere ``(-10, 60)``, and the room holding the television, and
never leaks outside. With the upstream default ``door=1`` (closed) the maze
splits into two components: the wing behind the sliding door contains the goal
at ``(-10, 60)`` and the start poses ``(-35, 40)`` and ``(-10, 40)``, so an
agent starting elsewhere has to press action ``4`` repeatedly to open the door
before it can ever reach the reward.

Deliberate differences from the Unity build
-------------------------------------------
1. Time is discrete: one environment step advances ``step_seconds`` seconds, so
   the forward action moves ``speed * step_seconds`` units. The Unity build is
   stepped by the physics loop, whose displacement per decision depends on
   ``Time.timeScale`` and cannot be recovered in closed form without running it.
2. Rotations use the upstream reset granularity of ``15`` degrees rather than
   the ``1.5``-degree ``MoveRotation`` call that the physics loop amplifies.
3. Rendering is a vectorized first-person ray caster of the same geometry. The
   Unity build renders textured meshes; here surfaces receive flat colours with
   distance shading.
4. The eight television textures shipped as ``Assets/00*.jpg`` are replaced by
   deterministic synthetic channel patterns, so no third-party image assets are
   redistributed. The channel *dynamics* are reproduced: in the noisy condition
   a channel is drawn uniformly at random every step, and the "one channel
   change per ten presses" rule is kept.

Using the original Unity build
------------------------------
:mod:`environments.noisy_tv_unity` wraps the upstream python client
(``unityagents``, shipped in the same repository) and drives the official
binary that project distributes, exposing it as ``NoisyTVUnity-v0``. It is
interchangeable with this module for the training pipeline: same observation
shape, same six actions, same sparse reward.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as error:  # pragma: no cover - gymnasium is a hard dependency
    raise ImportError(
        "NoisyTVMaze requires gymnasium; install it with "
        "`python -m pip install -r requirements-master.txt`"
    ) from error


ENVIRONMENT_ID = "NoisyTVMaze-v0"
"""Gymnasium registration ID of the pure-python reconstruction."""

ENTRY_POINT = "environments.noisy_tv_maze:NoisyTVMazeEnv"
"""Import path Gymnasium resolves for :data:`ENVIRONMENT_ID`."""

# ---------------------------------------------------------------------------
# Upstream scene geometry.
# ---------------------------------------------------------------------------

#: The maze's walls as axis-aligned footprints ``(x, z, half_x, half_z)`` in
#: world units, read from the 138 ``Cube`` objects of ``Assets/tv_maze.unity``
#: (local transforms composed up the scene hierarchy; rotated cubes reduced to
#: their axis-aligned footprint). Every wall is five units tall, matching the
#: upstream prefabs' ``y`` extent of ``[0, 5]``.
WALL_BOXES: Tuple[Tuple[float, float, float, float], ...] = (
    (0, -25.5, 6, 0.5),
    (20, -25.5, 6, 0.5),
    (5.5, -24.25, 0.5, 1.75),
    (14.5, -24.25, 0.5, 1.75),
    (10, -23, 5, 0.5),
    (-5.5, -20, 0.5, 6),
    (25.5, -20, 0.5, 6),
    (10, -17, 5, 0.5),
    (5.5, -15.75, 0.5, 1.75),
    (14.5, -15.75, 0.5, 1.75),
    (-4.25, -14.5, 1.75, 0.5),
    (4.25, -14.5, 1.75, 0.5),
    (15.75, -14.5, 1.75, 0.5),
    (24.25, -14.5, 1.75, 0.5),
    (-3, -10, 0.5, 5),
    (3, -10, 0.5, 5),
    (17, -10, 0.5, 5),
    (23, -10, 0.5, 5),
    (-20, -5.5, 6, 0.5),
    (-4.25, -5.5, 1.75, 0.5),
    (4.25, -5.5, 1.75, 0.5),
    (15.75, -5.5, 1.75, 0.5),
    (24.25, -5.5, 1.75, 0.5),
    (40, -5.5, 6, 0.5),
    (-25.5, -4.25, 0.5, 1.75),
    (-14.5, -4.25, 0.5, 1.75),
    (-5.5, -4.25, 0.5, 1.75),
    (25.5, -4.25, 0.5, 1.75),
    (34.5, -4.25, 0.5, 1.75),
    (-10, -3, 5, 0.5),
    (30, -3, 5, 0.5),
    (-31.5, -2.97, 6.5, 0.5),
    (5.5, 0, 0.5, 6),
    (14.5, 0, 0.5, 6),
    (45.5, 0, 0.5, 6),
    (-28.75, 3, 3.75, 0.5),
    (-10, 3, 5, 0.5),
    (30, 3, 5, 0.5),
    (-25.5, 4.25, 0.5, 1.75),
    (-14.5, 4.25, 0.5, 1.75),
    (-5.5, 4.25, 0.5, 1.75),
    (25.5, 4.25, 0.5, 1.75),
    (34.5, 4.25, 0.5, 1.75),
    (-24.25, 5.5, 1.75, 0.5),
    (-15.75, 5.5, 1.75, 0.5),
    (0, 5.5, 6, 0.5),
    (15.75, 5.5, 1.75, 0.5),
    (24.25, 5.5, 1.75, 0.5),
    (35.75, 5.5, 1.75, 0.5),
    (44.25, 5.5, 1.75, 0.5),
    (-38, 5.75, 0.5, 9.25),
    (-32, 8.75, 0.5, 6.25),
    (-23, 10, 0.5, 5),
    (-17, 10, 0.5, 5),
    (17, 10, 0.5, 5),
    (23, 10, 0.5, 5),
    (37, 10, 0.5, 5),
    (43, 10, 0.5, 5),
    (-55, 14.5, 6, 0.5),
    (-39.25, 14.5, 1.8, 0.55),
    (-30.75, 14.5, 1.8, 0.55),
    (-24.25, 14.5, 1.75, 0.5),
    (-15.75, 14.5, 1.75, 0.5),
    (0, 14.5, 6, 0.5),
    (15.75, 14.5, 1.75, 0.5),
    (24.25, 14.5, 1.75, 0.5),
    (35.75, 14.5, 1.75, 0.5),
    (44.25, 14.5, 1.75, 0.5),
    (-49.5, 15.75, 0.5, 1.75),
    (-40.5, 15.75, 0.5, 1.75),
    (5.5, 15.75, 0.5, 1.75),
    (14.5, 15.75, 0.5, 1.75),
    (25.5, 15.75, 0.5, 1.75),
    (34.5, 15.75, 0.5, 1.75),
    (-45, 17, 5, 0.5),
    (10, 17, 5, 0.5),
    (30, 17, 5, 0.5),
    (-60.5, 20, 0.5, 6),
    (-29.5, 20, 0.5, 6),
    (-25.5, 20, 0.5, 6),
    (-14.5, 20, 0.5, 6),
    (-5.5, 20, 0.5, 6),
    (45.5, 20, 0.5, 6),
    (-45, 23, 5, 0.5),
    (10, 23, 5, 0.5),
    (30, 23, 5, 0.5),
    (-49.5, 24.25, 0.5, 1.75),
    (-40.5, 24.25, 0.5, 1.75),
    (5.5, 24.25, 0.5, 1.75),
    (14.5, 24.25, 0.5, 1.75),
    (25.5, 24.25, 0.5, 1.75),
    (34.5, 24.25, 0.5, 1.75),
    (-55, 25.5, 6, 0.5),
    (-39.25, 25.5, 1.7, 0.51),
    (-30.75, 25.5, 1.7, 0.51),
    (-20, 25.5, 6, 0.5),
    (0, 25.5, 6, 0.5),
    (20, 25.5, 6, 0.5),
    (35.75, 25.5, 1.75, 0.5),
    (44.25, 25.5, 1.75, 0.5),
    (-38, 30, 0.5, 5),
    (-32, 30, 0.5, 5),
    (37, 30, 0.5, 5),
    (43, 30, 0.5, 5),
    (-39.25, 34.5, 1.75, 0.5),
    (-30.75, 34.5, 1.75, 0.5),
    (-10, 34.5, 6, 0.5),
    (20, 34.5, 6, 0.5),
    (35.75, 34.5, 1.75, 0.5),
    (44.25, 34.5, 1.75, 0.5),
    (-29.5, 35.75, 0.5, 1.75),
    (-15.5, 35.75, 0.5, 1.75),
    (25.5, 35.75, 0.5, 1.75),
    (34.5, 35.75, 0.5, 1.75),
    (-22.5, 37, 7.5, 0.5),
    (30, 37, 5, 0.5),
    (-40.5, 40, 0.5, 6),
    (-4.5, 40, 0.5, 6),
    (14.5, 40, 0.5, 6),
    (45.5, 40, 0.5, 6),
    (-22.5, 43, 7.5, 0.5),
    (30, 43, 5, 0.5),
    (-29.5, 44.25, 0.5, 1.75),
    (-15.5, 44.25, 0.5, 1.75),
    (25.5, 44.25, 0.5, 1.75),
    (34.5, 44.25, 0.5, 1.75),
    (-35, 45.5, 6, 0.5),
    (-14.25, 45.5, 1.75, 0.5),
    (-5.75, 45.5, 1.75, 0.5),
    (20, 45.5, 6, 0.5),
    (40, 45.5, 6, 0.5),
    (-13, 50, 0.5, 5),
    (-7, 50, 0.5, 5),
    (-14.25, 54.5, 1.75, 0.5),
    (-5.75, 54.5, 1.75, 0.5),
    (-15.5, 60, 0.5, 6),
    (-4.5, 60, 0.5, 6),
    (-10, 65.5, 6, 0.5),
)

#: Bounds of the upstream floor plane (``Floor`` at ``(5, 0, 20)`` scaled
#: ``160 x 100``): ``(x_min, x_max, z_min, z_max)``. Everything outside is
#: treated as wall, so rays always terminate.
FLOOR_BOUNDS = (-75.0, 85.0, -30.0, 70.0)

#: Room centres ``(x, z)`` of the upstream scene's 17 ``Room`` prefabs, kept for
#: documentation and for layout-consistency tests.
ROOM_CENTRES: Tuple[Tuple[float, float], ...] = (
    (-10.0, 60.0),
    (0.0, -20.0),
    (-35.0, 40.0),
    (-55.0, 20.0),
    (-20.0, 20.0),
    (-20.0, 0.0),
    (20.0, 20.0),
    (40.0, 20.0),
    (40.0, 40.0),
    (-10.0, 40.0),
    (0.0, 0.0),
    (0.0, 20.0),
    (20.0, 0.0),
    (20.0, 40.0),
    (20.0, -20.0),
    (40.0, 0.0),
    (-35.0, 20.0),
)

#: Hallway centres ``(x, z, yaw_degrees)`` of the upstream scene's 18 ``Hallway``
#: prefabs. Each prefab spans 10 units along its local x axis and five units
#: wide inside; chained pairs form the corridors between rooms.
HALLWAY_CENTRES: Tuple[Tuple[float, float, float], ...] = (
    (10.0, -20.0, 0.0),
    (-35.0, 30.0, 90.0),
    (0.0, -10.0, 90.0),
    (-30.0, 0.0, 0.0),
    (-20.0, 10.0, 90.0),
    (40.0, 30.0, 90.0),
    (40.0, 10.0, 90.0),
    (30.0, 0.0, 0.0),
    (30.0, 20.0, 0.0),
    (10.0, 20.0, 0.0),
    (30.0, 40.0, 0.0),
    (-10.0, 0.0, 0.0),
    (20.0, 10.0, 90.0),
    (20.0, -10.0, 90.0),
    (-10.0, 50.0, 90.0),
    (-35.0, 10.0, 90.0),
    (-45.0, 20.0, 0.0),
    (-20.0, 40.0, 0.0),
)

WALL_HEIGHT = 5.0
"""Wall height in world units (upstream cubes span ``y`` in ``[0, 5]``)."""

AGENT_EYE_HEIGHT = 1.5
"""Camera height in world units (upstream agent state uses ``y = 1.5``)."""

#: Upstream start poses ``(x, z, yaw_degrees)`` from ``TVAgent.InitializeAgent``.
START_POSES: Tuple[Tuple[float, float, float], ...] = (
    (20.0, 40.0, 90.0),
    (40.0, 40.0, 180.0),
    (40.0, 20.0, 180.0),
    (20.0, 20.0, 180.0),
    (0.0, 20.0, 90.0),
    (40.0, 0.0, -90.0),
    (20.0, 0.0, 180.0),
    (20.0, -20.0, -90.0),
    (0.0, -20.0, 180.0),
    (0.0, 0.0, -90.0),
    (-20.0, 0.0, -90.0),
    (-20.0, 20.0, 180.0),
    (-35.0, 20.0, 0.0),
    (-55.0, 20.0, 90.0),
    (-35.0, 40.0, 90.0),
    (-10.0, 40.0, 0.0),
)

GOAL_POSITION = (-10.0, 60.0)
"""Centre of the goal sphere (upstream ``Sphere`` at ``(-10, 2, 60)``)."""

GOAL_RADIUS = 1.0
"""Radius of the goal sphere (the unit sphere scaled by two)."""

GOAL_REACH_DISTANCE = 2.5
"""Upstream success test: ``distance(goal, agent) < 2.5`` pays ``+1``."""

TELEVISION_POSITION = (45.02, 39.97)
"""Television plane centre (upstream ``TV`` at ``x = 45.02, z = 39.97``)."""

TELEVISION_FACING = -1.0
"""The screen normal points towards ``-x``: it is watched from inside the room."""

TELEVISION_HALF_WIDTH = 4.5
"""Half width of the nine-unit-wide screen, measured along ``z``."""

TELEVISION_BOTTOM = 0.8
"""Lowest screen row in world units (centre ``2.8`` minus half height ``2``)."""

TELEVISION_TOP = 4.8
"""Highest screen row in world units."""

TELEVISION_INTERACTION_DISTANCE = 18.0
"""Upstream range within which action ``5`` may change the channel."""

DOOR_POSITION = (-35.5, 25.5)
"""Centre of the sliding door (upstream ``Sliding_Door`` object)."""

DOOR_HALF_WIDTH = 3.5
"""Half width of the door slab, measured along ``x`` (upstream scale ``7``)."""

DOOR_HALF_THICKNESS = 0.55
"""Half thickness of the door slab, measured along ``z`` (upstream ``1.1``)."""

BUTTON_PERIOD = 10
"""Upstream ``count = (count + 1) % 10``: buttons only act every tenth press."""

BUTTON_COUNTER_INITIAL = 3
"""Upstream ``count`` starts at three, so the first actuation needs seven presses."""

DOOR_MODES = ("open", "deterministic", "random")
"""``door`` conditions reproducing the upstream branches ``0.0``, ``1.0``, other."""

TELEVISION_MODES = ("static", "noisy", "interactive")
"""``tv`` conditions reproducing upstream ``0.0``, ``1.0`` and ``2.0``."""

_DOOR_CODE_TO_MODE = {0.0: "open", 1.0: "deterministic", 2.0: "random"}
_TELEVISION_CODE_TO_MODE = {0.0: "static", 1.0: "noisy", 2.0: "interactive"}

_FREE_STYLE = 0
_WALL_STYLE = 1
_DOOR_STYLE = 2

_CEILING_COLOUR = np.array([24, 24, 32], dtype=np.float64)
_FLOOR_COLOUR = np.array([58, 58, 66], dtype=np.float64)
_WALL_COLOUR = np.array([148, 148, 158], dtype=np.float64)
_DOOR_COLOUR = np.array([96, 138, 188], dtype=np.float64)
_GOAL_COLOUR = np.array([236, 196, 92], dtype=np.float64)
_FOG_DISTANCE = 45.0

_BOXES = np.asarray(WALL_BOXES, dtype=np.float64)
_GRID_CACHE: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]] = {}


def _normalise_mode(value: Any, modes: Tuple[str, ...], name: str) -> str:
    """Convert an upstream numeric code or a mode name into a canonical name.

    Input: Raw ``door``/``tv`` value (a name or the upstream float code), the
        accepted mode names, and the parameter name for error messages.
    Output: One of ``modes``.
    Mathematical meaning: Selects which upstream reset branch is reproduced
        (see ``TemplateAcademy.AcademyReset``).
    """
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in modes:
            return candidate
        raise ValueError(f"{name} must be one of {modes} or an upstream code, got {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        table = _DOOR_CODE_TO_MODE if name == "door" else _TELEVISION_CODE_TO_MODE
        mode = table.get(float(value))
        if mode is not None:
            return mode
        raise ValueError(f"{name}={value!r} is not an upstream code; use one of {modes}")
    raise TypeError(f"{name} must be a string or a number, got {type(value).__name__}")


def _build_grids(
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    """Rasterize the upstream walls and the sliding door on a regular grid.

    Input: Cell size in world units.
    Output: ``(wall, style, door, origin)`` where ``wall`` marks cells blocked
        by a wall, ``style`` carries :data:`_WALL_STYLE` there, ``door`` marks
        the door slab, and ``origin`` is the world coordinate of the ``(0, 0)``
        cell.
    Mathematical meaning: Discretizes the maze's obstacle set for ray casting;
        walls are the upstream footprints, and cells outside the floor plane
        count as wall so that every ray terminates.
    """
    x_min, x_max, z_min, z_max = FLOOR_BOUNDS
    xs = np.arange(x_min, x_max + resolution, resolution)
    zs = np.arange(z_min, z_max + resolution, resolution)
    grid_x, grid_z = np.meshgrid(xs + resolution / 2.0, zs + resolution / 2.0)

    inside_floor = (
        (grid_x >= x_min) & (grid_x <= x_max) & (grid_z >= z_min) & (grid_z <= z_max)
    )
    wall = ~inside_floor
    for box_x, box_z, half_x, half_z in WALL_BOXES:
        wall |= (np.abs(grid_x - box_x) <= half_x) & (np.abs(grid_z - box_z) <= half_z)
    door = (np.abs(grid_x - DOOR_POSITION[0]) <= DOOR_HALF_WIDTH) & (
        np.abs(grid_z - DOOR_POSITION[1]) <= DOOR_HALF_THICKNESS
    )
    wall &= ~door
    style = np.where(wall, _WALL_STYLE, _FREE_STYLE).astype(np.uint8)
    return wall, style, door, (x_min, z_min)


def _grids(resolution: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    """Return the cached rasterization for ``resolution``.

    Input: Cell size in world units.
    Output: The tuple produced by :func:`_build_grids`.
    Mathematical meaning: The obstacle discretization is a property of the
        upstream scene, so it is built once per resolution and shared by every
        environment instance.
    """
    if resolution not in _GRID_CACHE:
        _GRID_CACHE[resolution] = _build_grids(resolution)
    return _GRID_CACHE[resolution]


class NoisyTVMazeEnv(gym.Env):
    """Navigate the upstream TV maze, reach the goal, and (optionally) watch TV.

    The reward is sparse: ``+1`` on the transition that brings the agent within
    :data:`GOAL_REACH_DISTANCE` of the goal sphere, which also terminates the
    episode. Nothing else pays, the television included: the television only
    makes the observation stream *unpredictable*, which is exactly the
    phenomenon this environment exists to exhibit.

    Args:
        image_size: Side length of the square first-person RGB frame.
        max_steps: Truncation horizon in environment steps; the upstream
            ``tv_test.py`` driver loops a thousand steps per episode.
        start_loc: Upstream ``startLoc``. ``0`` samples one of
            :data:`START_POSES` uniformly and pairs it with a random heading
            that is a multiple of 15 degrees; ``1..16`` selects that pose.
        door: Upstream ``door``, as a mode name or the upstream numeric code.
            ``"open"``/``0.0`` opens the door and randomizes it, so every press
            toggles it with probability one half; ``"deterministic"``/``1.0``
            (the upstream scene default) closes it and makes every press act;
            ``"random"``/``2.0`` closes it and randomizes it.
        tv: Upstream ``tv``. ``"static"``/``0.0`` fixes the screen;
            ``"noisy"``/``1.0`` redraws it from a random channel every step
            (the paper's noisy television); ``"interactive"``/``2.0`` keeps it
            fixed until the agent presses action ``5`` within range.
        num_channels: Number of synthetic channels the screen can display.
        speed: Forward speed in world units per simulated second.
        step_seconds: Simulated seconds advanced by one environment step.
        turn_degrees: Heading change applied by one rotate action.
        agent_radius: Collision radius used against the walls.
        field_of_view: Horizontal field of view of the camera, in degrees.
        ray_resolution: Cell size in world units of the ray-casting grid.
        render_mode: ``None`` or ``"rgb_array"``, following the Gymnasium API.

    The environment is deterministic given a seed: the only random quantities
    are the start pose, the television channel, and door toggling in the
    randomized door modes.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        image_size: int = 84,
        max_steps: int = 1000,
        start_loc: int = 0,
        door: Any = 1.0,
        tv: Any = 1.0,
        num_channels: int = 8,
        speed: float = 5.0,
        step_seconds: float = 0.25,
        turn_degrees: float = 15.0,
        agent_radius: float = 0.5,
        field_of_view: float = 60.0,
        ray_resolution: float = 0.5,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the maze, its spaces, and its derived obstacle grids.

        Input: Frame size, horizon, upstream reset parameters, dynamics, and
            rendering settings.
        Output: An uninitialized environment; :meth:`reset` starts an episode.
        Mathematical meaning: Defines the MDP state (pose, television channel,
            door state), the discrete action set, the sparse reward, and the
            observation map from state to first-person RGB frames.
        """
        super().__init__()
        if int(image_size) < 16:
            raise ValueError("image_size must be at least 16 pixels")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be at least 1")
        if not 0 <= int(start_loc) <= len(START_POSES):
            raise ValueError(f"start_loc must lie in [0, {len(START_POSES)}]")
        if int(num_channels) < 2:
            raise ValueError("num_channels must be at least 2")
        if float(speed) <= 0.0 or float(step_seconds) <= 0.0:
            raise ValueError("speed and step_seconds must be positive")
        if not 0.0 < float(turn_degrees) < 180.0:
            raise ValueError("turn_degrees must lie in (0, 180)")
        if float(agent_radius) <= 0.0:
            raise ValueError("agent_radius must be positive")
        if not 0.0 < float(field_of_view) < 179.0:
            raise ValueError("field_of_view must lie in (0, 179)")
        if float(ray_resolution) <= 0.0:
            raise ValueError("ray_resolution must be positive")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be None or one of {self.metadata['render_modes']}")

        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.start_loc = int(start_loc)
        self.door_mode = _normalise_mode(door, DOOR_MODES, "door")
        self.television_mode = _normalise_mode(tv, TELEVISION_MODES, "tv")
        self.num_channels = int(num_channels)
        self.speed = float(speed)
        self.step_seconds = float(step_seconds)
        self.turn_degrees = float(turn_degrees)
        self.agent_radius = float(agent_radius)
        self.field_of_view = float(field_of_view)
        self.ray_resolution = float(ray_resolution)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )

        self._wall_grid, self._style_grid, self._door_grid, self._grid_origin = _grids(
            self.ray_resolution
        )
        self._ring_angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        self._channels = self._build_channels()

        # Episode state; reset() overwrites all of it.
        self._x = 0.0
        self._z = 0.0
        self._heading = 0.0
        self._channel = 0
        self._door_closed = True
        self._door_random = False
        self._button_counter = BUTTON_COUNTER_INITIAL
        self._steps = 0
        self._terminated = False
        self._truncated = False

    def _build_channels(self) -> np.ndarray:
        """Build the synthetic channel patterns shown on the television.

        Input: This environment's channel count.
        Output: ``uint8`` array shaped ``(num_channels, 4, 4, 3)``.
        Mathematical meaning: Defines the finite set of images the screen can
            display, standing in for the upstream texture list. What the paper
            studies is how these are *drawn over time*, not their content, so
            procedurally generated patterns preserve the phenomenon.
        """
        generator = np.random.default_rng(0)
        return generator.integers(0, 256, size=(self.num_channels, 4, 4, 3), dtype=np.uint8)

    # ------------------------------------------------------------ Gymnasium
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new episode at an upstream start pose.

        Input: Optional seed and an options dictionary that may override
            ``start_loc``, ``door``, and ``tv`` for this episode, mirroring the
            upstream ``env.reset(config={...})`` call.
        Output: ``(observation, info)`` for the initial state.
        Mathematical meaning: Samples the initial state ``s_0`` (pose, channel,
            door state) from the seeded randomness described by the reset
            parameters.
        """
        super().reset(seed=seed)
        options = options or {}
        start_loc = int(options.get("start_loc", self.start_loc))
        door_mode = _normalise_mode(options.get("door", self.door_mode), DOOR_MODES, "door")
        television_mode = _normalise_mode(
            options.get("tv", self.television_mode), TELEVISION_MODES, "tv"
        )
        if not 0 <= start_loc <= len(START_POSES):
            raise ValueError(f"start_loc must lie in [0, {len(START_POSES)}]")

        if start_loc == 0:
            index = int(self.np_random.integers(0, len(START_POSES)))
            self._x, self._z, _ = START_POSES[index]
            self._heading = float(self.np_random.integers(0, 24)) * 15.0
        else:
            self._x, self._z, self._heading = START_POSES[start_loc - 1]

        # Upstream door branch: 0 opens and randomizes, 1 closes and makes every
        # press act, otherwise closes and randomizes.
        self._door_closed = door_mode != "open"
        self._door_random = door_mode != "deterministic"
        self._button_counter = BUTTON_COUNTER_INITIAL
        self._steps = 0
        self._terminated = False
        self._truncated = False
        self._channel = (
            int(self.np_random.integers(0, self.num_channels))
            if television_mode == "noisy"
            else 0
        )
        return self._render(), self._info()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply one maze action and return the normalized transition.

        Input: Discrete action in ``0..5`` with the upstream meanings: ``1``
            forward, ``2`` rotate left, ``3`` rotate right, ``4`` press the
            sliding door, ``5`` change the television channel.
        Output: ``(observation, reward, terminated, truncated, info)``.
        Mathematical meaning: Applies the transition ``s_{t+1} = f(s_t, a_t)``
            (locomotion blocked by walls), draws a new channel when the
            television is noisy, and evaluates the sparse reward
            ``r_t = 1[||agent - goal|| < 2.5]`` with absorbing success.
        """
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after the episode ended; call reset() first")
        action_value = self._as_action(action)
        if not self.action_space.contains(action_value):
            raise ValueError(f"action must be an element of {self.action_space}, got {action!r}")

        if action_value == 1:
            self._advance(self.speed * self.step_seconds)
        elif action_value == 2:
            self._heading = (self._heading + self.turn_degrees) % 360.0
        elif action_value == 3:
            self._heading = (self._heading - self.turn_degrees) % 360.0
        elif action_value == 4:
            self._press_door()
        elif action_value == 5:
            self._press_channel()

        if self.television_mode == "noisy":
            self._channel = int(self.np_random.integers(0, self.num_channels))

        self._steps += 1
        success = self._distance_to_goal() < GOAL_REACH_DISTANCE
        self._terminated = bool(success)
        self._truncated = bool(not success and self._steps >= self.max_steps)
        reward = 1.0 if success else 0.0

        info = self._info()
        if self._truncated:
            info["TimeLimit.truncated"] = True
        return self._render(), reward, self._terminated, self._truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Return the current first-person frame.

        Input: This environment.
        Output: ``uint8`` array shaped ``(image_size, image_size, 3)``.
        Mathematical meaning: Evaluates the observation map ``o_t = O(s_t)``,
            including the television screen whenever it is in view.

        The frame is always recomputed from the current state; nothing is
        cached, so the observation can never disagree with the state (for
        example after :meth:`restore_state`).
        """
        return self._render()

    def close(self) -> None:
        """Release resources; the environment holds none.

        Input: This environment.
        Output: No value.
        Mathematical meaning: None.
        """
        return None

    # ------------------------------------------------------------ dynamics
    def _as_action(self, action: Any) -> int:
        """Convert an array-like action into a Python integer.

        Input: Scalar, NumPy scalar, or single-element array-like action.
        Output: Integer action value.
        Mathematical meaning: Extracts the categorical action ``a_t``.
        """
        array = np.asarray(action)
        if array.size != 1:
            raise ValueError(f"action must be a single discrete value, got {action!r}")
        try:
            return int(array.reshape(-1)[0])
        except (TypeError, ValueError) as error:
            raise ValueError(f"action must be integral, got {action!r}") from error

    def _point_is_free(self, x: float, z: float) -> bool:
        """Test whether a world point lies outside every wall footprint.

        Input: Candidate world position.
        Output: ``True`` when the point is inside the maze and outside the walls.
        Mathematical meaning: Membership of the configuration-space obstacle
            complement, which is what the Unity build's rigidbody collisions
            amount to for a point agent.
        """
        x_min, x_max, z_min, z_max = FLOOR_BOUNDS
        if not (x_min <= x <= x_max and z_min <= z <= z_max):
            return False
        delta_x = np.abs(x - _BOXES[:, 0])
        delta_z = np.abs(z - _BOXES[:, 1])
        if self._door_closed:
            if abs(x - DOOR_POSITION[0]) <= DOOR_HALF_WIDTH and (
                abs(z - DOOR_POSITION[1]) <= DOOR_HALF_THICKNESS
            ):
                return False
        return not bool(np.any((delta_x <= _BOXES[:, 2]) & (delta_z <= _BOXES[:, 3])))

    def _disc_is_free(self, x: float, z: float) -> bool:
        """Test whether a disc of radius ``agent_radius`` fits at ``(x, z)``.

        Input: Candidate world position.
        Output: ``True`` when the whole disc lies in free space.
        Mathematical meaning: Approximates the configuration-space obstacle
            complement by sampling the disc centre and a ring of twelve points.
        """
        points_x = x + self.agent_radius * np.cos(self._ring_angles)
        points_z = z + self.agent_radius * np.sin(self._ring_angles)
        points_x = np.append(points_x, x)
        points_z = np.append(points_z, z)
        x_min, x_max, z_min, z_max = FLOOR_BOUNDS
        if (
            points_x.min() < x_min
            or points_x.max() > x_max
            or points_z.min() < z_min
            or points_z.max() > z_max
        ):
            return False
        delta_x = np.abs(points_x[:, None] - _BOXES[None, :, 0])
        delta_z = np.abs(points_z[:, None] - _BOXES[None, :, 1])
        blocked = ((delta_x <= _BOXES[None, :, 2]) & (delta_z <= _BOXES[None, :, 3])).any()
        if blocked:
            return False
        if self._door_closed:
            door_blocked = (
                (np.abs(points_x - DOOR_POSITION[0]) <= DOOR_HALF_WIDTH)
                & (np.abs(points_z - DOOR_POSITION[1]) <= DOOR_HALF_THICKNESS)
            ).any()
            if door_blocked:
                return False
        return True

    def _advance(self, distance: float) -> None:
        """Drive forward, sliding along walls when the direct path is blocked.

        Input: Forward distance in world units.
        Output: No value; the agent position is updated in place.
        Mathematical meaning: Applies the upstream ``rb.velocity = forward * 5``
            move, projected onto the free configuration space so that the agent
            slides along walls instead of penetrating them, as the Unity
            rigidbody does.
        """
        radians = math.radians(self._heading)
        delta_x = math.sin(radians) * distance
        delta_z = math.cos(radians) * distance
        if self._disc_is_free(self._x + delta_x, self._z + delta_z):
            self._x += delta_x
            self._z += delta_z
            return
        if self._disc_is_free(self._x + delta_x, self._z):
            self._x += delta_x
            return
        if self._disc_is_free(self._x, self._z + delta_z):
            self._z += delta_z

    def _press_door(self) -> None:
        """Apply upstream action ``4``: press the sliding door button.

        Input: This environment's button counter and door state.
        Output: No value; every tenth press toggles the door as upstream
            ``SlidingDoor.press`` does.
        Mathematical meaning: ``count <- (count + 1) mod 10`` and, at zero, a
            toggle that always succeeds when the door is deterministic and
            succeeds with probability one half when it is randomized.
        """
        self._button_counter = (self._button_counter + 1) % BUTTON_PERIOD
        if self._button_counter != 0:
            return
        if (not self._door_random) or float(self.np_random.random()) > 0.5:
            self._door_closed = not self._door_closed

    def _press_channel(self) -> None:
        """Apply upstream action ``5``: change the television channel.

        Input: This environment's television mode, distance to the screen, and
            button counter.
        Output: No value; within :data:`TELEVISION_INTERACTION_DISTANCE` of the
            screen, every tenth press draws a channel uniformly at random.
        Mathematical meaning: Reproduces the upstream branch that assigns a
            random texture to the television material. The channel is the only
            state change, so the television never pays reward.
        """
        if self.television_mode == "static":
            return
        if self._distance_to_television() >= TELEVISION_INTERACTION_DISTANCE:
            return
        self._button_counter = (self._button_counter + 1) % BUTTON_PERIOD
        if self._button_counter != 0:
            return
        self._channel = int(self.np_random.integers(0, self.num_channels))

    def _distance_to_goal(self) -> float:
        """Return the distance from the agent to the goal sphere.

        Input: This environment's agent position.
        Output: Distance in world units.
        Mathematical meaning: The quantity compared with
            :data:`GOAL_REACH_DISTANCE` to decide success.
        """
        return math.hypot(self._x - GOAL_POSITION[0], self._z - GOAL_POSITION[1])

    def _distance_to_television(self) -> float:
        """Return the distance from the agent to the television.

        Input: This environment's agent position.
        Output: Distance in world units.
        Mathematical meaning: The quantity the upstream ``18.0`` interaction
            range is compared against.
        """
        return math.hypot(self._x - TELEVISION_POSITION[0], self._z - TELEVISION_POSITION[1])

    def _info(self) -> Dict[str, Any]:
        """Describe the current episode state.

        Input: This environment.
        Output: Dictionary with pose, distances to goal and television,
            television channel and mode, door state, step count, and success.
        Mathematical meaning: Reports the MDP state variables the upstream
            brain exposed as ``states`` plus the diagnostics the thesis monitors
            (distance to the reward, distance to the distractor).
        """
        return {
            "agent_position": (float(self._x), float(self._z)),
            "agent_heading": float(self._heading),
            "goal_position": GOAL_POSITION,
            "television_position": TELEVISION_POSITION,
            "distance_to_goal": self._distance_to_goal(),
            "distance_to_television": self._distance_to_television(),
            "television_channel": int(self._channel),
            "television_mode": self.television_mode,
            "door_closed": bool(self._door_closed),
            "steps": int(self._steps),
            "is_success": bool(self._distance_to_goal() < GOAL_REACH_DISTANCE),
        }

    # ------------------------------------------------------ simulator state
    def clone_state(self) -> Dict[str, Any]:
        """Snapshot the full episode state.

        Input: This environment.
        Output: Dictionary with pose, channel, door state, counter, and time.
        Mathematical meaning: Materializes ``s_t`` so episodic memory can
            return to a previously visited high-novelty state.
        """
        return {
            "x": float(self._x),
            "z": float(self._z),
            "heading": float(self._heading),
            "channel": int(self._channel),
            "door_closed": bool(self._door_closed),
            "door_random": bool(self._door_random),
            "button_counter": int(self._button_counter),
            "steps": int(self._steps),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
        }

    def restore_state(self, snapshot: Dict[str, Any]) -> np.ndarray:
        """Restore a snapshot produced by :meth:`clone_state`.

        Input: Dictionary returned by :meth:`clone_state`.
        Output: Observation of the restored state.
        Mathematical meaning: Replaces the current state by the stored ``s_t``
            without touching the reward function or the environment RNG.
        """
        self._x = float(snapshot["x"])
        self._z = float(snapshot["z"])
        self._heading = float(snapshot["heading"])
        self._channel = int(snapshot["channel"])
        self._door_closed = bool(snapshot["door_closed"])
        self._door_random = bool(snapshot["door_random"])
        self._button_counter = int(snapshot["button_counter"])
        self._steps = int(snapshot["steps"])
        self._terminated = bool(snapshot["terminated"])
        self._truncated = bool(snapshot["truncated"])
        return self._render()

    def get_obs(self) -> np.ndarray:
        """Return the current observation without stepping.

        Input: This environment.
        Output: Current first-person frame.
        Mathematical meaning: Reads ``o_t = O(s_t)``, used by the
            episodic-memory restoration path of Algorithm 2.
        """
        return self.render()

    # ----------------------------------------------------------- rendering
    def _render(self) -> np.ndarray:
        """Render the first-person view of the maze.

        Input: This environment's agent pose, door state, and channel.
        Output: ``uint8`` frame shaped ``(image_size, image_size, 3)``.
        Mathematical meaning: Evaluates the observation map: a pinhole camera
            at height :data:`AGENT_EYE_HEIGHT` with horizontal field of view
            ``field_of_view`` casts one ray per column and projects the walls,
            the sliding door, the television plane, and the goal sphere.
        """
        size = self.image_size
        columns = np.arange(size, dtype=np.float64)
        # Ray angles follow the Unity yaw convention used by the upstream start
        # poses: a heading of yaw points along ``(sin(yaw), cos(yaw))`` in the
        # ``(x, z)`` plane.
        angle = (
            math.radians(self._heading)
            + ((columns + 0.5) / size - 0.5) * math.radians(self.field_of_view)
        )
        distance, style = self._cast_rays(angle)

        # Perpendicular distances keep flat surfaces flat across the frame.
        angle_offset = angle - math.radians(self._heading)
        perpendicular = np.maximum(distance * np.cos(angle_offset), 1.0e-3)

        projection = (size / 2.0) / math.tan(math.radians(self.field_of_view) / 2.0)
        centre = (size - 1) / 2.0
        rows = np.arange(size, dtype=np.float64)[:, None]
        wall_top = centre - (WALL_HEIGHT - AGENT_EYE_HEIGHT) * projection / perpendicular
        wall_bottom = centre + AGENT_EYE_HEIGHT * projection / perpendicular
        wall_mask = (rows >= wall_top[None, :]) & (rows < wall_bottom[None, :])

        # Floor and ceiling: vertical gradients that darken away from the eye.
        vertical = np.abs(rows - centre) / max(centre, 1.0)
        row_colour = np.where(
            rows < centre,
            _CEILING_COLOUR[None, :] * (1.0 - 0.35 * vertical),
            _FLOOR_COLOUR[None, :] * (1.0 - 0.55 * vertical),
        )
        frame = np.repeat(row_colour[:, None, :], size, axis=1)

        # Walls and the door: flat colours with fog towards the far distance.
        shade = np.clip(1.0 - perpendicular / _FOG_DISTANCE, 0.25, 1.0)
        wall_colour = np.where(
            (style == _DOOR_STYLE)[:, None], _DOOR_COLOUR[None, :], _WALL_COLOUR[None, :]
        )
        wall_pixels = wall_colour * (0.65 * shade[:, None] + 0.35)
        frame = np.where(wall_mask[:, :, None], wall_pixels[None, :, :], frame)

        self._draw_television(frame, angle, perpendicular, projection, centre)
        self._draw_goal(frame, perpendicular)
        return np.clip(frame, 0.0, 255.0).astype(np.uint8)

    def _cast_rays(self, angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """March one ray per column through the wall grid (vectorized DDA).

        Input: Per-column ray angles in radians.
        Output: ``(distance, style)`` with the ray distance to the first
            blocking cell and the style code of that cell (wall or door).
        Mathematical meaning: Solves, per column, the first-intersection
            problem between a ray and the discretized obstacle set, which is
            what defines occlusion in the rendered image.
        """
        direction_x = np.sin(angle)
        direction_z = np.cos(angle)
        grid = self._wall_grid
        door_grid = self._door_grid
        origin_x, origin_z = self._grid_origin
        resolution = self.ray_resolution
        height_cells, width_cells = grid.shape
        ray_count = angle.shape[0]

        map_x = np.full(ray_count, int(math.floor((self._x - origin_x) / resolution)))
        map_z = np.full(ray_count, int(math.floor((self._z - origin_z) / resolution)))
        step_x = np.where(direction_x > 0.0, 1, -1)
        step_z = np.where(direction_z > 0.0, 1, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_x = np.where(direction_x != 0.0, np.abs(resolution / direction_x), np.inf)
            delta_z = np.where(direction_z != 0.0, np.abs(resolution / direction_z), np.inf)
            boundary_x = origin_x + (map_x + (step_x > 0)) * resolution
            boundary_z = origin_z + (map_z + (step_z > 0)) * resolution
            side_x = np.where(
                direction_x != 0.0, np.abs((boundary_x - self._x) / direction_x), np.inf
            )
            side_z = np.where(
                direction_z != 0.0, np.abs((boundary_z - self._z) / direction_z), np.inf
            )

        hit = np.zeros(ray_count, dtype=bool)
        distance = np.full(ray_count, _FOG_DISTANCE * 2.0)
        style = np.full(ray_count, _WALL_STYLE, dtype=np.uint8)
        travelled = np.zeros(ray_count)
        max_iterations = 4 * (width_cells + height_cells)
        for _ in range(max_iterations):
            if hit.all():
                break
            advance_x = side_x < side_z
            active = ~hit
            map_x = np.where(active & advance_x, map_x + step_x, map_x)
            map_z = np.where(active & ~advance_x, map_z + step_z, map_z)
            reached = np.where(advance_x, side_x, side_z)
            travelled = np.where(active, reached, travelled)
            side_x = np.where(active & advance_x, side_x + delta_x, side_x)
            side_z = np.where(active & ~advance_x, side_z + delta_z, side_z)
            inside = (
                (map_x >= 0) & (map_x < width_cells) & (map_z >= 0) & (map_z < height_cells)
            )
            blocked = np.zeros(ray_count, dtype=bool)
            door_cell = np.zeros(ray_count, dtype=bool)
            if inside.any():
                rows_in = map_z[inside]
                columns_in = map_x[inside]
                blocked[inside] = grid[rows_in, columns_in]
                if self._door_closed:
                    door_cell[inside] = door_grid[rows_in, columns_in]
            escaped = active & ~inside
            now_hit = active & (blocked | door_cell | escaped)
            hit_distance = np.where(escaped, travelled, reached)
            distance = np.where(now_hit, hit_distance, distance)
            style = np.where(
                now_hit, np.where(door_cell, _DOOR_STYLE, _WALL_STYLE).astype(np.uint8), style
            )
            hit |= now_hit
        return distance, style

    def _draw_television(
        self,
        frame: np.ndarray,
        angle: np.ndarray,
        perpendicular: np.ndarray,
        projection: float,
        centre: float,
    ) -> None:
        """Draw the television screen, including the channel it is showing.

        Input: Frame buffer, per-column ray angles and distances, and the
            projection constants.
        Output: No value; the frame is painted in place.
        Mathematical meaning: Projects the screen plane (normal along ``-x``)
            and samples the channel pattern, so the agent observes the
            unpredictable content the paper's noisy television injects into the
            observation stream.
        """
        direction_x = np.sin(angle)
        direction_z = np.cos(angle)
        plane_x = TELEVISION_POSITION[0]
        facing = direction_x * TELEVISION_FACING < 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            distance_along = np.where(
                np.abs(direction_x) > 1.0e-9, (plane_x - self._x) / direction_x, np.inf
            )
        hit_z = self._z + distance_along * direction_z
        visible = (
            facing
            & np.isfinite(distance_along)
            & (distance_along > 0.0)
            & (distance_along < perpendicular * 1.05)
            & (np.abs(hit_z - TELEVISION_POSITION[1]) <= TELEVISION_HALF_WIDTH)
        )
        if not visible.any():
            return
        rows = np.arange(frame.shape[0], dtype=np.float64)[:, None]
        top = centre - (TELEVISION_TOP - AGENT_EYE_HEIGHT) * projection / perpendicular
        bottom = centre - (TELEVISION_BOTTOM - AGENT_EYE_HEIGHT) * projection / perpendicular
        mask = visible[None, :] & (rows >= top[None, :]) & (rows < bottom[None, :])
        if not mask.any():
            return
        # Sample the channel pattern using normalised screen coordinates.
        row_index = np.clip(((rows - top) / np.maximum(bottom - top, 1.0) * 4.0), 0, 3.99)
        column_index = np.clip(
            (
                (hit_z - TELEVISION_POSITION[1] + TELEVISION_HALF_WIDTH)
                / (2.0 * TELEVISION_HALF_WIDTH)
                * 4.0
            ),
            0,
            3.99,
        )
        pattern = self._channels[self._channel]
        rows_tiled = np.broadcast_to(
            row_index.astype(np.int64) % 4, (frame.shape[0], frame.shape[1])
        )
        columns_tiled = np.broadcast_to(
            column_index.astype(np.int64)[None, :] % 4, (frame.shape[0], frame.shape[1])
        )
        screen = pattern[rows_tiled, columns_tiled].astype(np.float64)
        shade = np.clip(1.0 - perpendicular / _FOG_DISTANCE, 0.3, 1.0)
        screen = screen * (0.6 * shade[None, :] + 0.4)[:, :, None]
        frame[mask] = screen[mask]

    def _draw_goal(self, frame: np.ndarray, perpendicular: np.ndarray) -> None:
        """Draw the goal sphere as a billboard disc.

        Input: Frame buffer and per-column wall distances (for occlusion).
        Output: No value; the frame is painted in place.
        Mathematical meaning: Projects the centre and radius of the sphere that
            defines the task's reward, so the agent can see what it is looking
            for.
        """
        relative_x = GOAL_POSITION[0] - self._x
        relative_z = GOAL_POSITION[1] - self._z
        heading = math.radians(self._heading)
        # Unity convention: forward is ``(sin, cos)`` and right is its -90 degree
        # rotation, ``(cos, -sin)``, in the ``(x, z)`` plane.
        forward_x, forward_z = math.sin(heading), math.cos(heading)
        right_x, right_z = math.cos(heading), -math.sin(heading)
        depth = relative_x * forward_x + relative_z * forward_z
        if depth <= 0.2:
            return
        lateral = relative_x * right_x + relative_z * right_z
        size = frame.shape[0]
        projection = (size / 2.0) / math.tan(math.radians(self.field_of_view) / 2.0)
        centre_column = (
            (size - 1) / 2.0
            + (math.atan2(lateral, depth) / math.radians(self.field_of_view)) * size
        )
        column = int(round(centre_column))
        if 0 <= column < size:
            if depth * math.cos(math.atan2(lateral, depth)) > perpendicular[column] + 0.1:
                return
        radius_pixels = projection * GOAL_RADIUS / max(depth, 0.2)
        centre_row = (size - 1) / 2.0 - (2.0 - AGENT_EYE_HEIGHT) * projection / depth
        rows = np.arange(size)[:, None]
        columns = np.arange(size)[None, :]
        radius = max(radius_pixels, 1.0)
        circle = (((rows - centre_row) / radius) ** 2 + ((columns - centre_column) / radius) ** 2) <= 1.0
        if circle.any():
            shade = float(np.clip(1.0 - depth / _FOG_DISTANCE, 0.3, 1.0))
            colour = _GOAL_COLOUR * (0.55 * shade + 0.45)
            frame[circle] = np.broadcast_to(colour, (frame.shape[0], frame.shape[1], 3))[circle]

    # --------------------------------------------------------- connectivity
    def reachable_cells(self, door_closed: Optional[bool] = None, resolution: float = 1.0):
        """Flood fill the free space from the agent's current position.

        Input: Optional door state override (defaults to the current one) and
            the grid resolution of the fill.
        Output: ``(mask, origin, resolution)`` with the reachable cells marked.
        Mathematical meaning: Computes the connected component of free space
            the agent can occupy, which lets tests (and users) verify that the
            goal and the television room are actually reachable through the
            upstream door layout.
        """
        block_door = self._door_closed if door_closed is None else bool(door_closed)
        x_min = FLOOR_BOUNDS[0]
        z_min = FLOOR_BOUNDS[2]
        xs = np.arange(x_min, FLOOR_BOUNDS[1] + resolution, resolution)
        zs = np.arange(z_min, FLOOR_BOUNDS[3] + resolution, resolution)
        grid_x, grid_z = np.meshgrid(xs, zs)
        free = np.ones(grid_x.shape, dtype=bool)
        for box_x, box_z, half_x, half_z in WALL_BOXES:
            free &= ~((np.abs(grid_x - box_x) <= half_x) & (np.abs(grid_z - box_z) <= half_z))
        if block_door:
            free &= ~(
                (np.abs(grid_x - DOOR_POSITION[0]) <= DOOR_HALF_WIDTH)
                & (np.abs(grid_z - DOOR_POSITION[1]) <= DOOR_HALF_THICKNESS)
            )
        start_x = int(round((self._x - x_min) / resolution))
        start_z = int(round((self._z - z_min) / resolution))
        mask = np.zeros_like(free)
        if not (0 <= start_z < free.shape[0] and 0 <= start_x < free.shape[1]):
            return mask, (x_min, z_min), resolution
        if not free[start_z, start_x]:
            return mask, (x_min, z_min), resolution
        mask[start_z, start_x] = True
        for _ in range(4 * (free.shape[0] + free.shape[1])):
            grown = (
                mask
                | np.roll(mask, 1, axis=0)
                | np.roll(mask, -1, axis=0)
                | np.roll(mask, 1, axis=1)
                | np.roll(mask, -1, axis=1)
            ) & free
            if np.array_equal(grown, mask):
                break
            mask = grown
        return mask, (x_min, z_min), resolution


def register_environment(environment_id: str = ENVIRONMENT_ID) -> str:
    """Register the Noisy-TV maze with Gymnasium once.

    Input: Registration ID, defaulting to :data:`ENVIRONMENT_ID`.
    Output: The registered ID.
    Mathematical meaning: Makes the MDP addressable by name so that
        ``gym.make(environment_id)`` instantiates the same task as
        :class:`NoisyTVMazeEnv`.

    An existing registration under the same ID is preserved, so a user-provided
    environment keeps precedence.
    """
    registry = getattr(gym, "registry", None) or gym.envs.registry
    if environment_id not in registry:
        gym.register(
            id=environment_id,
            entry_point=ENTRY_POINT,
            reward_threshold=None,
            disable_env_checker=False,
        )
    return environment_id
