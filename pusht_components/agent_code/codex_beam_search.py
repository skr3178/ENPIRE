"""Codex (GPT-5.5) agent-written Push-T policy.

Approach: beam search over action-only push primitives. The policy plans in a
PRIVATE simulator, caches the target-action sequence, and returns one target per
real env step. It does not mutate the live environment.

Provenance: recovered verbatim from the ENPIRE website JS bundle
(_next/static/chunks/0aoq0j8a~y3.9.js). The only edit is filling the one
gym.make(...) call that was truncated at a template-interpolation seam:
    gym.make("gym_pusht/PushT-v0", obs_type="state")   # <- filled (was blank)
Everything else is the Codex agent's original code.

Note: this policy REQUIRES env access (it reads env.unwrapped internals) and so
expects to be called as policy(obs, env=<the live env>).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

import gym_pusht  # noqa: F401 - registers env


def _wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return np.asarray([1.0, 0.0] if fallback is None else fallback, dtype=np.float64)
    return vector / norm


State = tuple[np.ndarray, tuple[float, float], tuple[float, float], float]


@dataclass
class BeamSearchPushTPolicy:
    beam_width: int = 32
    max_depth: int = 4
    workspace_low: float = 0.0
    workspace_high: float = 512.0
    success_bonus: float = 10000.0
    _sim_env: gym.Env | None = field(default=None, init=False, repr=False)
    _planned_actions: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _action_index: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._planned_actions = []
        self._action_index = 0
        if self._sim_env is not None:
            self._sim_env.close()
            self._sim_env = None

    def __call__(self, observation: np.ndarray | list[float] | tuple[float, ...], env: Any | None = None) -> np.ndarray:
        if env is None:
            return np.asarray(observation, dtype=np.float64)[2:4].astype(np.float32)
        if self._action_index >= len(self._planned_actions):
            self._planned_actions = self._plan(env.unwrapped)
            self._action_index = 0
        action = self._planned_actions[self._action_index]
        self._action_index += 1
        return action.astype(np.float32)

    def act(self, observation: np.ndarray | list[float] | tuple[float, ...], env: Any | None = None) -> np.ndarray:
        return self(observation, env=env)

    def _ensure_sim(self) -> gym.Env:
        if self._sim_env is None:
            self._sim_env = gym.make("gym_pusht/PushT-v0", obs_type="state")  # <- filled seam
            self._sim_env.reset(seed=999)
        return self._sim_env

    def _plan(self, real: Any) -> list[np.ndarray]:
        sim_env = self._ensure_sim()
        sim = sim_env.unwrapped
        goal_pose = np.asarray(real.goal_pose, dtype=np.float64)
        root_state = self._capture_state(real)
        root_score = self._score_state(real, goal_pose)[0]
        beam: list[tuple[float, State, list[np.ndarray]]] = [(root_score, root_state, [])]
        best_plan: list[np.ndarray] = []

        for _ in range(self.max_depth):
            candidates: list[tuple[float, State, list[np.ndarray], bool]] = []
            for _, state, plan in beam:
                self._restore_state(sim, state)
                for primitive in self._primitives(sim, goal_pose):
                    self._restore_state(sim, state)
                    terminated = False
                    for action in primitive:
                        _, _, terminated, _, _ = sim_env.step(action)
                        if terminated:
                            break
                    score, _, _, _ = self._score_state(sim, goal_pose)
                    if terminated:
                        score += self.success_bonus
                    candidates.append((score, self._capture_state(sim), plan + primitive, terminated))

            candidates.sort(key=lambda item: item[0], reverse=True)
            if not candidates:
                break
            best_plan = candidates[0][2]
            if candidates[0][3]:
                return best_plan
            beam = [(score, state, plan) for score, state, plan, _ in candidates[: self.beam_width]]

        return best_plan or [np.asarray(real.get_obs(), dtype=np.float64)[2:4].astype(np.float32)]

    def _capture_state(self, env_unwrapped: Any) -> State:
        return (
            np.asarray(env_unwrapped.get_obs(), dtype=np.float64).copy(),
            tuple(np.asarray(env_unwrapped.agent.velocity, dtype=np.float64)),
            tuple(np.asarray(env_unwrapped.block.velocity, dtype=np.float64)),
            float(env_unwrapped.block.angular_velocity),
        )

    def _restore_state(self, env_unwrapped: Any, state: State) -> None:
        obs, agent_velocity, block_velocity, block_angular_velocity = state
        env_unwrapped._set_state(obs)
        env_unwrapped._set_state(obs)
        env_unwrapped.agent.velocity = agent_velocity
        env_unwrapped.block.velocity = block_velocity
        env_unwrapped.block.angular_velocity = block_angular_velocity
        env_unwrapped.n_contact_points = 0

    def _score_state(self, env_unwrapped: Any, goal_pose: np.ndarray) -> tuple[float, float, float, float]:
        coverage = float(env_unwrapped._get_coverage())
        block_pose = np.asarray(env_unwrapped.get_obs(), dtype=np.float64)[2:5]
        position_error = float(np.linalg.norm(block_pose[:2] - goal_pose[:2]))
        angle_error = abs(float(_wrap_pi(block_pose[2] - goal_pose[2])))
        score = coverage * 100.0 - position_error * 0.25 - angle_error * 20.0
        return score, coverage, position_error, angle_error

    def _primitives(self, env_unwrapped: Any, goal_pose: np.ndarray) -> list[list[np.ndarray]]:
        obs = np.asarray(env_unwrapped.get_obs(), dtype=np.float64)
        block_xy = obs[2:4]
        block_angle = float(obs[4])
        keypoints = np.asarray(env_unwrapped.get_keypoints(env_unwrapped._block_shapes), dtype=np.float64)
        contact_points = list(keypoints)
        for start, end in ((0, 1), (1, 2), (2, 5), (5, 6), (6, 7), (7, 4), (4, 3), (3, 0)):
            contact_points.append(0.5 * (keypoints[start] + keypoints[end]))

        to_goal = _unit(goal_pose[:2] - block_xy, np.array([np.cos(goal_pose[2]), np.sin(goal_pose[2])]))
        angle_error = _wrap_pi(goal_pose[2] - block_angle)
        spin = np.sign(angle_error) or 1.0

        primitives: list[list[np.ndarray]] = []
        for contact in contact_points:
            radius = contact - block_xy
            tangent = _unit(np.array([-radius[1], radius[0]]), np.array([0.0, 1.0]))
            directions = [
                to_goal,
                -to_goal,
                tangent,
                -tangent,
                _unit(to_goal + tangent, to_goal),
                _unit(to_goal - tangent, to_goal),
                _unit(0.4 * to_goal + spin * tangent, tangent),
                _unit(0.4 * to_goal - spin * tangent, -tangent),
            ]
            for direction in directions:
                for push_length in (120.0, 180.0, 240.0):
                    actions = [contact - 60.0 * direction] * 5
                    actions += [contact - 20.0 * direction] * 2
                    actions += [contact + push_length * direction] * 14
                    primitives.append([self._clip(action) for action in actions])
        return primitives

    def _clip(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.workspace_low, self.workspace_high).astype(np.float32)


def act(observation: np.ndarray | list[float] | tuple[float, ...], env: Any | None = None) -> np.ndarray:
    return BeamSearchPushTPolicy()(observation, env=env)
