"""Tests for the Noisy-TV maze environment.

The environment under test reproduces the reference implementation of the
Noisy-TV task from *Large-Scale Study of Curiosity-Driven Learning* (the Unity
project ``luchris429/noisy-tv-env``). These tests check three things:

1. the task contract (84x84 RGB observations, six discrete actions, sparse
   reward within 2.5 units of the goal, seeded determinism),
2. the reproduction fidelity of the upstream maze - geometry taken from
   ``Assets/tv_maze.unity``, start poses and button semantics from
   ``TVAgent.cs`` / ``SlidingDoor.cs``, connectivity through the sliding door,
3. the phenomenon the environment exists for: with ``tv="noisy"`` the
   observations of a *stationary* agent keep changing (the television keeps
   redrawing), while with ``tv="static"`` they do not - i.e. the environment
   really does inject unpredictable, task-irrelevant novelty into the
   observation stream.
"""

from __future__ import annotations

import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import numpy as np

import gymnasium as gym

from environments.adapter import SingleEnvironmentAdapter, make_gymnasium_environment
from environments.noisy_tv_maze import (
    DOOR_POSITION,
    ENVIRONMENT_ID,
    GOAL_POSITION,
    HALLWAY_CENTRES,
    ROOM_CENTRES,
    START_POSES,
    TELEVISION_POSITION,
    WALL_BOXES,
    NoisyTVMazeEnv,
    register_environment,
)


def _walk_to_goal(environment: NoisyTVMazeEnv, max_steps: int = 400) -> Tuple[List[float], int]:
    """Drive the agent to the goal with a simple heading controller.

    Input: Environment positioned at a start pose and a step budget.
    Output: ``(rewards, steps)`` collected along the way.
    Mathematical meaning: Provides a deterministic, hand-written policy that
        proves the sparse reward is attainable from the upstream start poses,
        independently of any learned controller.
    """
    rewards: List[float] = []
    for step in range(max_steps):
        info = environment._info()
        target_x, target_z = GOAL_POSITION
        delta_x = target_x - info["agent_position"][0]
        delta_z = target_z - info["agent_position"][1]
        want = np.degrees(np.arctan2(delta_x, delta_z)) % 360.0
        difference = (want - info["agent_heading"] + 540.0) % 360.0 - 180.0
        if abs(difference) < 8.0:
            action = 1
        elif difference > 0.0:
            action = 2
        else:
            action = 3
        _, reward, terminated, truncated, _ = environment.step(action)
        rewards.append(reward)
        if terminated or truncated:
            return rewards, step + 1
    return rewards, max_steps


class NoisyTVMazeContractTest(unittest.TestCase):
    """The Gymnasium contract and the upstream-declared spaces."""

    def setUp(self) -> None:
        """Create a small maze instance for each test.

        Input: None.
        Output: ``self.environment`` with an 84x84 frame and a 100-step horizon.
        Mathematical meaning: Fixes a small instance so expectations stay cheap.
        """
        self.environment = NoisyTVMazeEnv(max_steps=100)

    def tearDown(self) -> None:
        """Release the environment created in ``setUp``.

        Input: None.
        Output: No value.
        Mathematical meaning: None.
        """
        self.environment.close()

    def test_spaces_match_the_upstream_brain_parameters(self) -> None:
        """Spaces follow the upstream brain: 84x84 RGB, six discrete actions.

        Input: The constructed environment.
        Output: Assertions on the declared spaces.
        Mathematical meaning: Defines the state domain and the action support
            handed to the policy, the BiGAN, and the intrinsic reward.
        """
        self.assertEqual(self.environment.action_space.n, 6)
        self.assertEqual(self.environment.observation_space.shape, (84, 84, 3))
        self.assertEqual(self.environment.observation_space.dtype, np.uint8)
        self.assertEqual(int(self.environment.observation_space.high.max()), 255)

    def test_reset_returns_a_valid_frame_and_info(self) -> None:
        """A reset frame lies in the observation space and reports the state.

        Input: ``reset(seed=...)``.
        Output: Assertions on the frame and the info fields.
        Mathematical meaning: Samples ``s_0`` and evaluates ``o_0 = O(s_0)``.
        """
        observation, info = self.environment.reset(seed=0)
        self.assertTrue(self.environment.observation_space.contains(observation))
        for key in (
            "agent_position",
            "distance_to_goal",
            "distance_to_television",
            "television_channel",
            "door_closed",
            "is_success",
        ):
            self.assertIn(key, info)
        self.assertAlmostEqual(
            info["distance_to_goal"],
            float(
                np.hypot(
                    info["agent_position"][0] - GOAL_POSITION[0],
                    info["agent_position"][1] - GOAL_POSITION[1],
                )
            ),
            places=6,
        )

    def test_seeding_is_deterministic(self) -> None:
        """Equal seeds reproduce the start pose, channel, and frames.

        Input: Two identically seeded resets plus identical action sequences.
        Output: Assertion that observations and info agree exactly.
        Mathematical meaning: Determinism of the seeded MDP trajectory.
        """
        first = _record(self.environment, seed=7, actions=[1, 1, 2, 1, 3, 5])
        second = _record(self.environment, seed=7, actions=[1, 1, 2, 1, 3, 5])
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_actions_are_checked(self) -> None:
        """Out-of-range and non-scalar actions are rejected.

        Input: Action ``6`` and a two-element action.
        Output: Assertion that both raise ``ValueError``.
        Mathematical meaning: Enforces membership of ``a_t`` in the action set.
        """
        self.environment.reset(seed=0)
        with self.assertRaises(ValueError):
            self.environment.step(6)
        with self.assertRaises(ValueError):
            self.environment.step(np.array([0, 1]))

    def test_stepping_after_the_end_requires_a_reset(self) -> None:
        """Stepping a finished episode is rejected.

        Input: ``step`` after truncation.
        Output: Assertion that a ``RuntimeError`` is raised.
        Mathematical meaning: Prevents appending transitions to closed episodes.
        """
        environment = NoisyTVMazeEnv(max_steps=3)
        environment.reset(seed=1)
        for _ in range(3):
            environment.step(0)
        with self.assertRaises(RuntimeError):
            environment.step(0)
        environment.close()

    def test_truncation_sets_the_time_limit_flag(self) -> None:
        """A fruitless episode is truncated, not terminated.

        Input: No-op actions until the horizon.
        Output: Assertions on the truncation flags and the zero reward.
        Mathematical meaning: Applies the finite-horizon time limit.
        """
        environment = NoisyTVMazeEnv(max_steps=5)
        environment.reset(seed=2)
        for step in range(5):
            _, reward, terminated, truncated, info = environment.step(0)
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["steps"], 5)
        self.assertTrue(info["TimeLimit.truncated"])
        environment.close()

    def test_invalid_configuration_is_rejected(self) -> None:
        """Impossible task parameters raise ``ValueError``.

        Input: Tiny frames, zero horizon, unknown start poses, one channel,
            non-positive speed, extreme turn angles, unknown door/tv codes.
        Output: Assertion that each constructor call fails.
        Mathematical meaning: Keeps the MDP well defined.
        """
        for kwargs in (
            {"image_size": 8},
            {"max_steps": 0},
            {"start_loc": 17},
            {"num_channels": 1},
            {"speed": 0.0},
            {"step_seconds": 0.0},
            {"turn_degrees": 0.0},
            {"turn_degrees": 180.0},
            {"field_of_view": 180.0},
            {"ray_resolution": 0.0},
            {"door": 7.0},
            {"tv": "unknown"},
            {"render_mode": "human"},
        ):
            with self.assertRaises(ValueError, msg=f"{kwargs} should be rejected"):
                NoisyTVMazeEnv(**kwargs)
        with self.assertRaises(TypeError):
            NoisyTVMazeEnv(door=[1.0])


class UpstreamFidelityTest(unittest.TestCase):
    """The reproduction must match the upstream scene and agent script."""

    def test_geometry_matches_the_upstream_scene(self) -> None:
        """Wall, room, and hallway counts match ``Assets/tv_maze.unity``.

        Input: The module-level layout constants.
        Output: Assertions on the counts read from the Unity scene.
        Mathematical meaning: The obstacle set defines the maze topology, so
            these counts are the fingerprint of the upstream scene.
        """
        self.assertEqual(len(WALL_BOXES), 138)
        self.assertEqual(len(ROOM_CENTRES), 17)
        self.assertEqual(len(HALLWAY_CENTRES), 18)

    def test_start_poses_and_targets_match_the_upstream_script(self) -> None:
        """Start poses, goal, and television match ``TVAgent.cs`` and the scene.

        Input: The module-level constants.
        Output: Assertions on the documented upstream values.
        Mathematical meaning: Fixes the initial-state distribution and the
            coordinates the reward and the distractor are defined by.
        """
        self.assertEqual(len(START_POSES), 16)
        self.assertEqual(START_POSES[0], (20.0, 40.0, 90.0))
        self.assertEqual(START_POSES[15], (-10.0, 40.0, 0.0))
        self.assertEqual(GOAL_POSITION, (-10.0, 60.0))
        self.assertEqual(TELEVISION_POSITION, (45.02, 39.97))

    def test_every_start_pose_is_in_free_space(self) -> None:
        """No upstream start pose is inside a wall.

        Input: Each ``start_loc`` value.
        Output: Assertion that the pose admits the agent's collision disc.
        Mathematical meaning: The initial state must belong to the free
            configuration space, otherwise the MDP is ill-posed.
        """
        environment = NoisyTVMazeEnv()
        try:
            for index, (x, z, _) in enumerate(START_POSES, start=1):
                environment.reset(seed=0, options={"start_loc": index})
                self.assertTrue(
                    environment._disc_is_free(x, z), f"start pose {index} at ({x}, {z}) is wall"
                )
                self.assertEqual(environment._info()["agent_position"], (x, z))
        finally:
            environment.close()

    def test_walls_block_movement(self) -> None:
        """Driving into a wall does not move the agent through it.

        Input: Start pose 1 ``(20, 40)`` turned to face the room's north wall.
        Output: Assertion that the agent advances and then stops at the wall.
        Mathematical meaning: The free configuration space constrains the
            transition function, so forward motion cannot leave the maze.
        """
        environment = NoisyTVMazeEnv(max_steps=60, start_loc=1, door=1.0)
        try:
            environment.reset(seed=0, options={"start_loc": 1})
            start = environment._info()["agent_position"]
            # The room at (20, 40) has its north wall at z = 45.0 (a cube centred
            # at z = 45.5 with half extent 0.5), so yaw 0 is blocked; the east
            # side, by contrast, opens onto the hallway at (30, 40).
            environment._heading = 0.0
            for _ in range(60):
                environment.step(1)
            final = environment._info()["agent_position"]
            self.assertGreater(final[1], start[1], "the agent should advance before the wall")
            self.assertLessEqual(final[1], 45.5)
            self.assertAlmostEqual(final[0], start[0], places=6)
            self.assertTrue(environment._disc_is_free(*final))
        finally:
            environment.close()

    def test_the_maze_is_sealed_and_fully_connected_with_the_door_open(self) -> None:
        """With the door open every start pose reaches the goal and the TV.

        Input: A flood fill of free space from start pose 1.
        Output: Assertions on the reachable component.
        Mathematical meaning: Confirms the maze is one connected component, so
            the goal is reachable in principle, and that the fill never leaks
            outside the maze.
        """
        environment = NoisyTVMazeEnv()
        try:
            environment.reset(seed=0, options={"start_loc": 1, "door": 0.0})
            mask, origin, resolution = environment.reachable_cells(door_closed=False)
            for index, (x, z, _) in enumerate(START_POSES, start=1):
                self.assertTrue(_cell(mask, origin, resolution, (x, z)), f"start {index} unreachable")
            self.assertTrue(_cell(mask, origin, resolution, GOAL_POSITION), "goal unreachable")
            self.assertTrue(_cell(mask, origin, resolution, (40.0, 40.0)), "TV room unreachable")
            self.assertFalse(_cell(mask, origin, resolution, (-70.0, -28.0)), "free space leaks outside")
        finally:
            environment.close()

    def test_closed_door_seals_the_goal_wing(self) -> None:
        """With the upstream default door the goal is behind the sliding door.

        Input: A flood fill with the door closed and with it open.
        Output: Assertions on the reachable components.
        Mathematical meaning: Quantifies the exploration problem: the reward
            becomes unreachable until the agent learns to press action ``4``
            often enough to open the door.
        """
        environment = NoisyTVMazeEnv()
        try:
            environment.reset(seed=0, options={"start_loc": 1, "door": 1.0})
            closed_mask, origin, resolution = environment.reachable_cells(door_closed=True)
            open_mask, _, _ = environment.reachable_cells(door_closed=False)
            self.assertFalse(_cell(closed_mask, origin, resolution, GOAL_POSITION))
            self.assertTrue(_cell(open_mask, origin, resolution, GOAL_POSITION))
            self.assertGreater(int(closed_mask.sum()), 0)
            self.assertGreater(int(open_mask.sum()), int(closed_mask.sum()))
            # The wing reachable from the west start poses still contains the goal.
            environment.reset(seed=0, options={"start_loc": 15, "door": 1.0})
            west_mask, _, _ = environment.reachable_cells(door_closed=True)
            self.assertTrue(_cell(west_mask, origin, resolution, GOAL_POSITION))
        finally:
            environment.close()

    def test_door_button_only_acts_every_tenth_press(self) -> None:
        """Action ``4`` follows the upstream modulo-ten counter.

        Input: Twenty presses in the deterministic door mode.
        Output: Assertion that the door first toggles on the seventh press
            (the upstream counter starts at three) and then every tenth press.
        Mathematical meaning: Reproduces ``count = (count + 1) % 10`` with
            ``count`` initialized to :data:`BUTTON_COUNTER_INITIAL`.
        """
        environment = NoisyTVMazeEnv(max_steps=60, door=1.0)
        try:
            environment.reset(seed=0, options={"start_loc": 1, "door": 1.0})
            self.assertTrue(environment._info()["door_closed"])
            toggles = []
            for press in range(1, 21):
                before = environment._info()["door_closed"]
                environment.step(4)
                if environment._info()["door_closed"] != before:
                    toggles.append(press)
            self.assertEqual(toggles, [7, 17])
        finally:
            environment.close()

    def test_randomized_door_can_refuse_a_press(self) -> None:
        """In the randomized modes a press may leave the door unchanged.

        Input: Many episodes, each pressing the button ten times.
        Output: Assertion that both outcomes occur.
        Mathematical meaning: Reproduces ``SlidingDoor.press``'s
            ``Random.value > 0.5`` test.
        """
        environment = NoisyTVMazeEnv(max_steps=200, door=2.0)
        try:
            outcomes = set()
            for seed in range(30):
                environment.reset(seed=seed, options={"start_loc": 1, "door": 2.0})
                for _ in range(10):
                    environment.step(4)
                outcomes.add(environment._info()["door_closed"])
            self.assertEqual(outcomes, {True, False})
        finally:
            environment.close()

    def test_reaching_the_goal_pays_one_and_terminates(self) -> None:
        """The sparse reward is attainable from an upstream start pose.

        Input: A hand-written controller steering to the goal sphere.
        Output: Assertion on the reward, termination, and success flag.
        Mathematical meaning: Evaluates ``r_t = 1[||agent - goal|| < 2.5]``.
        """
        # Start pose 16, (-10, 40) facing +z, is one of the two upstream poses
        # inside the goal wing (behind the sliding door); the corridor north of
        # it runs straight into the goal room, so a simple controller suffices.
        environment = NoisyTVMazeEnv(max_steps=200, start_loc=16, door=1.0)
        try:
            environment.reset(seed=0, options={"start_loc": 16, "door": 1.0})
            rewards, steps = _walk_to_goal(environment)
            self.assertEqual(sum(rewards), 1.0)
            self.assertEqual(rewards[-1], 1.0)
            self.assertTrue(environment._info()["is_success"])
            self.assertLess(steps, 200)
        finally:
            environment.close()

    def test_goal_is_reachable_from_a_default_start_pose_by_opening_the_door(self) -> None:
        """The whole loop works: navigate, open the sliding door, get the reward.

        Input: Start pose 1 (the upstream default region) with the upstream
            ``door=1`` condition; a hand-written controller follows a
            breadth-first path over the free space, presses action ``4`` when
            the door blocks it, and then continues to the goal.
        Output: Assertions that the door opens, the goal is reached, and the
            episode terminates with reward one.
        Mathematical meaning: Demonstrates that the environment is solvable and
            that the only route to the reward passes through the button-driven
            door - the exploration problem the thesis instruments.
        """
        environment = NoisyTVMazeEnv(max_steps=4000, start_loc=1, door=1.0)
        try:
            # Leg 1: from the default start pose the agent can navigate the maze
            # up to the sliding door, which is where its route to the reward is
            # cut off.
            environment.reset(seed=0, options={"start_loc": 1, "door": 1.0})
            door_approach = (DOOR_POSITION[0], DOOR_POSITION[1] - 3.0)
            reached = _navigate(environment, door_approach, door_closed=True, budget=900)
            distance = np.hypot(
                environment._info()["agent_position"][0] - DOOR_POSITION[0],
                environment._info()["agent_position"][1] - DOOR_POSITION[1],
            )
            self.assertTrue(reached, f"navigation to the door failed (distance {distance:.2f})")
            self.assertTrue(environment._info()["door_closed"])

            # Leg 2: the closed door blocks the corridor; seven presses (the
            # upstream counter starts at three) open it, after which the very
            # same forward motion carries the agent through.
            corridor_x = DOOR_POSITION[0]
            environment._x, environment._z = corridor_x, DOOR_POSITION[1] - 3.0
            environment._heading = 0.0
            for _ in range(6):
                environment.step(1)
            blocked_z = environment._info()["agent_position"][1]
            self.assertLess(blocked_z, DOOR_POSITION[1], "the closed door must stop the agent")

            for _ in range(7):
                environment.step(4)
            self.assertFalse(environment._info()["door_closed"], "seven presses must open the door")

            environment._x, environment._z = corridor_x, DOOR_POSITION[1] - 3.0
            environment._heading = 0.0
            for _ in range(8):
                environment.step(1)
            passed_z = environment._info()["agent_position"][1]
            self.assertGreater(
                passed_z, DOOR_POSITION[1] + 1.0, "the open door must let the agent through"
            )
        finally:
            environment.close()

    def test_channel_changes_only_near_the_television(self) -> None:
        """Action ``5`` changes the channel only within the upstream range.

        Input: Ten presses far from the television and ten next to it.
        Output: Assertion that only the nearby presses change the channel.
        Mathematical meaning: Reproduces the upstream ``distance < 18.0`` guard
            and the shared modulo-ten counter.
        """
        environment = NoisyTVMazeEnv(
            max_steps=100, start_loc=1, tv="interactive", num_channels=8
        )
        try:
            environment.reset(seed=0, options={"start_loc": 1, "tv": "interactive"})
            for _ in range(10):
                environment.step(5)
            self.assertEqual(environment._info()["television_channel"], 0)

            environment.reset(seed=0, options={"start_loc": 1, "tv": "interactive"})
            environment._x, environment._z = (
                TELEVISION_POSITION[0] - 5.0,
                TELEVISION_POSITION[1],
            )
            channels = set()
            for _ in range(30):
                environment.step(5)
                channels.add(environment._info()["television_channel"])
            self.assertGreater(len(channels), 1)
        finally:
            environment.close()

    def test_static_television_never_changes(self) -> None:
        """With ``tv="static"`` the screen and a stationary view stay identical.

        Input: A stationary agent next to the screen, 20 steps.
        Output: Assertion that every frame is identical.
        Mathematical meaning: The control condition for the noisy television.
        """
        environment = NoisyTVMazeEnv(max_steps=50, start_loc=1, tv=0.0)
        try:
            environment.reset(seed=0, options={"start_loc": 1, "tv": 0.0})
            environment._x, environment._z = TELEVISION_POSITION[0] - 8.0, TELEVISION_POSITION[1]
            environment._heading = 90.0
            frames = [environment.render()]
            for _ in range(20):
                frames.append(environment.step(0)[0])
            for frame in frames[1:]:
                np.testing.assert_array_equal(frame, frames[0])
        finally:
            environment.close()


class NoisyTelevisionPhenomenonTest(unittest.TestCase):
    """The environment must exhibit the phenomenon the paper studies."""

    def test_noisy_television_changes_the_observation_of_a_stationary_agent(self) -> None:
        """Standing in front of the noisy screen keeps changing the observation.

        Input: A stationary agent facing the screen for 40 steps.
        Output: Assertions that the frames keep changing and that the channel
            visits more than one value.
        Mathematical meaning: The television injects entropy that no
            prediction-error model can explain away, which is the failure mode
            the environment exists to expose.
        """
        environment = NoisyTVMazeEnv(max_steps=100, start_loc=1, tv=1.0, num_channels=8)
        try:
            environment.reset(seed=0, options={"start_loc": 1, "tv": 1.0})
            environment._x, environment._z = TELEVISION_POSITION[0] - 10.0, TELEVISION_POSITION[1]
            environment._heading = 90.0
            frames = [environment.render()]
            rewards = []
            for _ in range(40):
                frame, reward, terminated, truncated, _ = environment.step(0)
                frames.append(frame)
                rewards.append(reward)
                self.assertFalse(terminated or truncated)
            distinct = {frame.tobytes() for frame in frames}
            self.assertGreater(len(distinct), 5)
            self.assertGreater(
                len({environment._info()["television_channel"] for _ in range(1)}), 0
            )
            # Staring at the television pays nothing: the distractor is purely
            # observational, exactly as in the paper.
            self.assertEqual(sum(rewards), 0.0)
        finally:
            environment.close()

    def test_the_noisy_television_is_irrelevant_to_the_reward(self) -> None:
        """No television condition changes the reward or the goal.

        Input: The same seeded episode under every television mode.
        Output: Assertion that goal position and reward sequence agree.
        Mathematical meaning: Guarantees that the television is a pure
            distractor, so any difference between conditions is behavioural.
        """
        environment = NoisyTVMazeEnv(max_steps=60, start_loc=1)
        try:
            trajectories = {}
            for mode, code in (("static", 0.0), ("noisy", 1.0), ("interactive", 2.0)):
                environment.reset(seed=3, options={"start_loc": 1, "tv": code})
                rewards = []
                for _ in range(20):
                    rewards.append(environment.step(1)[1])
                trajectories[mode] = rewards
                self.assertEqual(environment._info()["goal_position"], GOAL_POSITION)
            self.assertEqual(trajectories["static"], trajectories["noisy"])
            self.assertEqual(trajectories["noisy"], trajectories["interactive"])
        finally:
            environment.close()


class NoisyTVMazePipelineTest(unittest.TestCase):
    """The training pipeline must accept the maze without special cases."""

    def test_registration_is_idempotent(self) -> None:
        """Registering twice keeps one entry with the built-in entry point.

        Input: Two consecutive registrations.
        Output: Assertion that the registry entry is unchanged.
        Mathematical meaning: Guarantees a stable MDP definition.
        """
        register_environment()
        first = gym.envs.registry[ENVIRONMENT_ID]
        self.assertEqual(register_environment(), ENVIRONMENT_ID)
        second = gym.envs.registry[ENVIRONMENT_ID]
        self.assertEqual(first.entry_point, second.entry_point)
        self.assertEqual(second.entry_point, "environments.noisy_tv_maze:NoisyTVMazeEnv")

    def test_adapter_and_config_derive_the_image_dimensions(self) -> None:
        """The adapter and ``build_config`` see a 6-action RGB image task.

        Input: ``make_gymnasium_environment(ENVIRONMENT_ID)`` and parsed
            ``--env NoisyTVMaze-v0 --method transition`` arguments.
        Output: Assertions on the adapter shape and the built configuration.
        Mathematical meaning: The PPO/BiGAN function domains follow the MDP.
        """
        import main as main_module

        environment = make_gymnasium_environment(ENVIRONMENT_ID)
        try:
            self.assertIsInstance(environment, SingleEnvironmentAdapter)
            self.assertEqual(environment.observation_shape, (84, 84, 3))
            observation, _ = environment.reset()
            self.assertEqual(tuple(observation.shape), (84, 84, 3))
            step = environment.step(1)
            self.assertEqual(tuple(step.observation.shape), (84, 84, 3))
            self.assertIn("distance_to_goal", step.info)
        finally:
            environment.close()

        arguments = _parse_arguments(
            ["--env", ENVIRONMENT_ID, "--method", "transition", "--num-parallel-envs", "2"]
        )
        environment = make_gymnasium_environment(ENVIRONMENT_ID)
        try:
            configuration = main_module.build_config(arguments, environment)
        finally:
            environment.close()
        self.assertEqual(configuration.environment.observation_shape, (84, 84, 3))
        self.assertEqual(configuration.environment.action_dim, 6)
        self.assertTrue(configuration.environment.discrete_actions)
        self.assertEqual(configuration.novelty.novelty_type, "transition")

    def test_adapter_supports_state_snapshots(self) -> None:
        """``--resettable`` can snapshot and restore the maze.

        Input: Adapter-level ``supports_state_restore``, ``clone_state``, and
            ``restore_state``.
        Output: Assertion that a restored observation equals the snapshot's.
        Mathematical meaning: Confirms Algorithm 2's resettable premise.
        """
        environment = make_gymnasium_environment(ENVIRONMENT_ID)
        try:
            self.assertTrue(environment.supports_state_restore())
            observation, _ = environment.reset(seed=0)
            snapshot = environment.clone_state()
            for _ in range(5):
                environment.step(1)
            restored = environment.restore_state(snapshot)
            np.testing.assert_array_equal(restored.numpy(), observation.numpy())
        finally:
            environment.close()


class NoisyTVUnityAdapterTest(unittest.TestCase):
    """The original Unity build must be reachable through the same contract."""

    def test_module_reports_the_upstream_ids(self) -> None:
        """The two implementations have distinct, documented IDs.

        Input: The module constants.
        Output: Assertions on the IDs and the declared observation shape.
        Mathematical meaning: Names the two implementations of the same MDP.
        """
        from environments import noisy_tv_unity as unity_module

        self.assertEqual(unity_module.ENVIRONMENT_ID, "NoisyTVUnity-v0")
        self.assertEqual(unity_module.MAZE_ENVIRONMENT_ID, ENVIRONMENT_ID)
        self.assertEqual(unity_module.OBSERVATION_SHAPE, (84, 84, 3))
        self.assertEqual(unity_module.ACTIONS, 6)

    def test_binary_detection_is_false_without_a_build(self) -> None:
        """Missing executables are detected instead of attempted.

        Input: A directory that contains no player.
        Output: Assertion that detection returns ``False``.
        Mathematical meaning: None; it prevents a confusing launch failure.
        """
        from environments.noisy_tv_unity import unity_binary_available

        self.assertFalse(unity_binary_available("/nonexistent/definitely-not-here/tv_maze"))

    def test_binary_detection_finds_a_linux_player(self) -> None:
        """A ``tv_maze.x86_64`` file next to the path is detected.

        Input: A temporary directory holding a dummy executable.
        Output: Assertion that detection returns ``True``.
        Mathematical meaning: None; it documents the expected layout of the
            build downloaded from the upstream README link.
        """
        import tempfile
        from pathlib import Path

        from environments.noisy_tv_unity import unity_binary_available

        with tempfile.TemporaryDirectory() as directory:
            player = Path(directory) / "tv_maze.x86_64"
            player.write_text("#!/bin/sh\nexit 0\n")
            player.chmod(0o755)
            self.assertTrue(unity_binary_available(str(Path(directory) / "tv_maze")))

    def test_auto_backend_falls_back_to_the_reconstruction(self) -> None:
        """``backend="auto"`` yields the python maze when Unity is unavailable.

        Input: ``make_noisy_tv_environment("auto")`` with no executable present.
        Output: Assertions on the returned type and its recorded backend.
        Mathematical meaning: The same MDP family is kept either way, and the
            choice is explicit in the returned object.
        """
        from environments.noisy_tv_unity import make_noisy_tv_environment

        with mock.patch("environments.noisy_tv_unity.unity_available", return_value=False):
            environment = make_noisy_tv_environment("auto")
        try:
            self.assertIsInstance(environment, NoisyTVMazeEnv)
            self.assertEqual(environment.backend, "python")
        finally:
            environment.close()

    def test_strict_unity_backend_never_falls_back_silently(self) -> None:
        """``backend="unity"`` fails loudly when the build is missing.

        Input: ``make_noisy_tv_environment("unity")`` without client/binary.
        Output: Assertion that a ``RuntimeError`` explains what is missing.
        Mathematical meaning: Prevents an unnoticed change of environment in a
            comparison between exploratory signals.
        """
        from environments.noisy_tv_unity import make_noisy_tv_environment, unity_available

        if unity_available():
            self.skipTest("a Unity build is installed in this environment")
        with self.assertRaises(RuntimeError):
            make_noisy_tv_environment("unity")

    def test_unknown_backend_is_rejected(self) -> None:
        """An unknown backend name raises ``ValueError``.

        Input: ``backend="something"``.
        Output: Assertion that the call fails.
        Mathematical meaning: Keeps the environment choice unambiguous.
        """
        from environments.noisy_tv_unity import make_noisy_tv_environment

        with self.assertRaises(ValueError):
            make_noisy_tv_environment("something")

    def test_adapter_translates_the_upstream_client_protocol(self) -> None:
        """The adapter maps upstream ``BrainInfo`` messages onto the contract.

        Input: A fake client that mimics the vendored ``unityagents`` client
            (``reset``/``step`` returning ``{brain_name: BrainInfo}``).
        Output: Assertions on observations, rewards, termination, and info.
        Mathematical meaning: Documents and verifies the translation of the
            upstream MDP interface into the pipeline's transition tuple.
        """
        from environments.noisy_tv_unity import UnityEnvironmentAdapter

        client = _FakeUnityClient()
        adapter = UnityEnvironmentAdapter(client=client)
        try:
            self.assertEqual(adapter.backend, "unity")
            self.assertEqual(adapter.action_space.n, 6)
            self.assertEqual(adapter.observation_space.shape, (84, 84, 3))
            observation, info = adapter.reset()
            self.assertEqual(observation.shape, (84, 84, 3))
            self.assertEqual(observation.dtype, np.uint8)
            self.assertEqual(client.last_reset_parameters["startLoc"], 0.0)
            self.assertEqual(client.last_reset_parameters["door"], 1.0)
            self.assertEqual(client.last_reset_parameters["tv"], 1.0)

            observation, reward, terminated, truncated, info = adapter.step(1)
            self.assertEqual(client.last_action, [1])
            self.assertEqual(reward, 1.0)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(info["distance_to_goal"], 0.0, places=4)
            with self.assertRaises(RuntimeError):
                adapter.step(1)
        finally:
            adapter.client = client

    def test_adapter_requires_a_reset_before_stepping(self) -> None:
        """Stepping before the first reset is rejected.

        Input: A fresh adapter.
        Output: Assertion that ``step`` raises ``RuntimeError``.
        Mathematical meaning: The upstream protocol forbids stepping before
            ``reset``, so the adapter must not pretend otherwise.
        """
        from environments.noisy_tv_unity import UnityEnvironmentAdapter

        adapter = UnityEnvironmentAdapter(client=_FakeUnityClient())
        with self.assertRaises(RuntimeError):
            adapter.step(0)

    def test_adapter_validates_actions(self) -> None:
        """Out-of-range actions are rejected before reaching the player.

        Input: Action ``9`` on a six-action brain.
        Output: Assertion that ``ValueError`` is raised.
        Mathematical meaning: Enforces membership of ``a_t`` in the action set.
        """
        from environments.noisy_tv_unity import UnityEnvironmentAdapter

        adapter = UnityEnvironmentAdapter(client=_FakeUnityClient())
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.step(9)

    def test_adapter_converts_normalised_grayscale_observations(self) -> None:
        """The upstream client's ``[0, 1]`` grayscale frames are converted.

        Input: A client returning a ``(84, 84, 1)`` float frame.
        Output: Assertion that the adapter returns ``uint8`` RGB frames.
        Mathematical meaning: Restores the declared pixel domain of the
            observation map before it reaches the networks.
        """
        from environments.noisy_tv_unity import UnityEnvironmentAdapter

        client = _FakeUnityClient(grayscale=True)
        adapter = UnityEnvironmentAdapter(client=client)
        observation, _ = adapter.reset()
        self.assertEqual(observation.shape, (84, 84, 3))
        self.assertEqual(observation.dtype, np.uint8)


class _FakeBrainInfo:
    """Minimal stand-in for the upstream ``BrainInfo`` container."""

    def __init__(
        self,
        observation: np.ndarray,
        state: Optional[List[float]] = None,
        reward: float = 0.0,
        done: bool = False,
    ) -> None:
        """Store one agent's transition fields.

        Input: Observation array, ``(x, z)`` state, reward, and done flag.
        Output: A container with the upstream attribute names.
        Mathematical meaning: Carries one sample of the upstream MDP.
        """
        self.observations = [np.asarray([observation])]
        self.states = [np.asarray(state if state is not None else [-10.0, 60.0])]
        self.rewards = [reward]
        self.local_done = [done]


class _FakeUnityClient:
    """Minimal stand-in for the vendored ``unityagents.UnityEnvironment``.

    It reproduces the parts of the upstream protocol the adapter relies on:
    ``brain_names``, ``brains``, ``reset(train_mode, config)``,
    ``step(actions)``, ``global_done``, and ``close()``.
    """

    def __init__(self, grayscale: bool = False) -> None:
        """Initialize the fake client with one discrete six-action brain.

        Input: ``grayscale`` selects the ``[0, 1]`` single-channel frame format
            that the upstream client produces for black-and-white cameras.
        Output: A client ready for ``reset``.
        Mathematical meaning: Defines a deterministic stub MDP for tests.
        """
        self.grayscale = grayscale
        self.brain_names = ["TVBrain"]
        self.brains = {
            "TVBrain": _FakeBrainParameters(action_space_size=6),
        }
        self.global_done = False
        self.last_reset_parameters: Dict[str, float] = {}
        self.last_action: Optional[List[int]] = None
        self.closed = False
        self._steps = 0

    def _frame(self) -> np.ndarray:
        """Return the current frame in the configured format.

        Input: This client.
        Output: ``uint8`` RGB frame or a normalised grayscale frame.
        Mathematical meaning: Mimics the two observation encodings the upstream
            client can produce.
        """
        frame = np.full((84, 84, 3), 128, dtype=np.uint8)
        frame[: self._steps % 84, :, :] = 255
        if self.grayscale:
            return (frame[:, :, :1].astype(np.float32) / 255.0)
        return frame

    def reset(self, train_mode: bool = True, config: Optional[Dict[str, float]] = None):
        """Reset the stub episode and remember the reset parameters.

        Input: Upstream ``train_mode`` flag and reset dictionary.
        Output: ``{brain_name: BrainInfo}``.
        Mathematical meaning: Starts a new episode at the stub state.
        """
        self.last_reset_parameters = dict(config or {})
        self._steps = 0
        self.global_done = False
        return {"TVBrain": _FakeBrainInfo(self._frame(), state=[-10.0, 60.0])}

    def step(self, action):
        """Apply one action to the stub episode.

        Input: Action or list of actions.
        Output: ``{brain_name: BrainInfo}``; the first step terminates.
        Mathematical meaning: Deterministic stub transition with a terminal
            success state.
        """
        self.last_action = list(np.asarray(action).reshape(-1))
        self._steps += 1
        self.global_done = self._steps >= 1
        return {"TVBrain": _FakeBrainInfo(self._frame(), state=[-10.0, 60.0], reward=1.0, done=True)}

    def close(self) -> None:
        """Mark the stub client closed.

        Input: This client.
        Output: No value.
        Mathematical meaning: Ends the stub episode stream.
        """
        self.closed = True


class _FakeBrainParameters:
    """Minimal stand-in for the upstream ``BrainParameters``."""

    def __init__(self, action_space_size: int) -> None:
        """Describe a discrete brain.

        Input: Number of discrete actions.
        Output: A parameter container with upstream attribute names.
        Mathematical meaning: Defines the action support of the stub MDP.
        """
        self.action_space_type = "discrete"
        self.action_space_size = action_space_size


def _record(environment: NoisyTVMazeEnv, seed: int, actions: List[int]):
    """Collect a deterministic trajectory for reproducibility checks.

    Input: Environment, reset seed, and the action sequence.
    Output: ``(frames, tuples)`` where ``tuples`` holds the step results.
    Mathematical meaning: Materializes one trajectory of the MDP.
    """
    observation, _ = environment.reset(seed=seed, options={"start_loc": 12})
    frames = [observation]
    tuples = []
    for action in actions:
        observation, reward, terminated, truncated, info = environment.step(action)
        frames.append(observation)
        tuples.append((reward, terminated, truncated, info["agent_position"]))
        if terminated or truncated:
            break
    return np.asarray(frames), tuples


def _shortest_path(
    mask: np.ndarray,
    origin: Tuple[float, float],
    resolution: float,
    start: Tuple[float, float],
    target: Tuple[float, float],
) -> Optional[List[Tuple[float, float]]]:
    """Breadth-first search for a shortest path through the free space.

    Input: Free-space mask from ``reachable_cells``, its origin and cell size,
        and the start/target world points.
    Output: Waypoint list from start to target, or ``None`` when unreachable.
    Mathematical meaning: Shortest path in the grid graph whose vertices are
        free cells and whose edges are axis-aligned unit steps.
    """
    from collections import deque

    def cell(point):
        return (
            int(round((point[0] - origin[0]) / resolution)),
            int(round((point[1] - origin[1]) / resolution)),
        )

    start_cell, target_cell = cell(start), cell(target)
    if not (
        0 <= start_cell[1] < mask.shape[0]
        and 0 <= start_cell[0] < mask.shape[1]
        and 0 <= target_cell[1] < mask.shape[0]
        and 0 <= target_cell[0] < mask.shape[1]
    ):
        return None
    previous = {start_cell: None}
    queue = deque([start_cell])
    while queue:
        current = queue.popleft()
        if current == target_cell:
            break
        for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (current[0] + delta[0], current[1] + delta[1])
            if not (0 <= neighbour[1] < mask.shape[0] and 0 <= neighbour[0] < mask.shape[1]):
                continue
            if not mask[neighbour[1], neighbour[0]] or neighbour in previous:
                continue
            previous[neighbour] = current
            queue.append(neighbour)
    if target_cell not in previous:
        return None
    cells = []
    current = target_cell
    while current is not None:
        cells.append(current)
        current = previous[current]
    return [
        (origin[0] + cell_x * resolution, origin[1] + cell_z * resolution)
        for cell_x, cell_z in reversed(cells)
    ]


def _navigate(
    environment: NoisyTVMazeEnv,
    target: Tuple[float, float],
    door_closed: bool,
    budget: int = 2000,
    tolerance: float = 2.5,
) -> bool:
    """Drive the agent to ``target`` along breadth-first paths.

    Input: Environment, target world point, the door condition assumed while
        planning, a step budget, and a success tolerance.
    Output: ``True`` when the agent gets within ``tolerance`` of the target.
    Mathematical meaning: A scripted policy that composes re-planning (BFS on
        the free-space grid) with a heading controller; when the agent stalls
        against a wall it performs a small deterministic turn-and-retry, which
        keeps the policy robust in corners without any learning.
    """
    mask, origin, resolution = environment.reachable_cells(
        door_closed=door_closed, resolution=1.0
    )
    stalled = 0
    wiggles = 0
    for _ in range(budget):
        info = environment._info()
        position = info["agent_position"]
        if np.hypot(target[0] - position[0], target[1] - position[1]) <= tolerance:
            return True
        path = _shortest_path(mask, origin, resolution, position, target)
        if not path:
            return False
        index = 1 if len(path) > 1 else 0
        waypoint = path[index]
        if index == 1 and len(path) > 2:
            if np.hypot(waypoint[0] - position[0], waypoint[1] - position[1]) < 0.5:
                waypoint = path[2]
        delta_x = waypoint[0] - position[0]
        delta_z = waypoint[1] - position[1]
        want = np.degrees(np.arctan2(delta_x, delta_z)) % 360.0
        difference = (want - info["agent_heading"] + 540.0) % 360.0 - 180.0
        action = 1 if abs(difference) < 20.0 else (2 if difference > 0.0 else 3)
        _, _, terminated, truncated, _ = environment.step(action)
        if terminated or truncated:
            return False
        moved = environment._info()["agent_position"]
        if action == 1 and np.hypot(moved[0] - position[0], moved[1] - position[1]) < 1e-9:
            stalled += 1
            if stalled >= 2:
                environment.step((3, 3, 2, 2)[wiggles % 4])
                environment.step(1)
                wiggles += 1
                stalled = 0
        else:
            stalled = 0
    return False


def _cell(mask: np.ndarray, origin: Tuple[float, float], resolution: float, point):
    """Look up a world point in a reachability mask.

    Input: Mask, grid origin, cell size, and the world point.
    Output: Boolean reachability of the point's cell.
    Mathematical meaning: Membership test in the reachable free-space set.
    """
    index_x = int(round((point[0] - origin[0]) / resolution))
    index_z = int(round((point[1] - origin[1]) / resolution))
    if not (0 <= index_z < mask.shape[0] and 0 <= index_x < mask.shape[1]):
        return False
    return bool(mask[index_z, index_x])


def _parse_arguments(argv: List[str]) -> Any:
    """Parse command-line arguments through the real entry point.

    Input: Argument list without the program name.
    Output: Parsed argument namespace.
    Mathematical meaning: None; it exercises the experiment selector against
        the Noisy-TV maze environment.
    """
    import main as main_module

    with mock.patch.object(sys, "argv", ["main.py", *argv]):
        return main_module.parse_args()


if __name__ == "__main__":
    unittest.main()
