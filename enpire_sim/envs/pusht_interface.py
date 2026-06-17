"""EN module: the autonomous environment interface a policy/agent may call.

This wraps the *immutable* ``gym_pusht/PushT-v0`` env and exposes exactly the
surface ENPIRE's Environment module provides:

* ``auto_reset(seed)`` -- select a randomized, valid initial state.
* ``step(action)`` -- apply a safety-clipped action under a step budget.
* coverage / success read-outs (verification signal).
* ``frame()`` -- an RGB frame for video logging.

The raw env code in ``gym-pusht/`` is treated as read-only (the paper's
"you are not allowed to modify environment code" rule). Everything here only
*calls* the public Gymnasium API.
"""
from __future__ import annotations

import enpire_sim  # noqa: F401 - sets SDL_VIDEODRIVER=dummy before pygame import

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np

import gym_pusht  # noqa: F401 - registers gym_pusht/PushT-v0

ENV_ID = "gym_pusht/PushT-v0"
ACTION_LOW, ACTION_HIGH = 0.0, 512.0


@dataclass
class EnvConfig:
    obs_type: str = "state"          # [agent_xy, block_xy, block_angle]
    max_steps: int = 300             # env's registered TimeLimit
    render: bool = False             # enable rgb_array frames for video


class PushTInterface:
    """Thin, safe, loggable handle on one Push-T 'robot station' (sim instance)."""

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        render_mode = "rgb_array" if self.config.render else None
        # max_episode_steps overrides the registered TimeLimit so the budget is explicit.
        self.env = gym.make(
            ENV_ID,
            obs_type=self.config.obs_type,
            render_mode=render_mode,
            max_episode_steps=self.config.max_steps,
        )
        self._t = 0
        self._last_coverage = 0.0
        # Source the success threshold FROM the env -- gym-pusht is the single
        # source of truth for reward/success (we never redefine it).
        self.success_threshold = float(self.env.unwrapped.success_threshold)

    # -- lifecycle -----------------------------------------------------------
    def auto_reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset to a randomized valid initial state (env handles overlap checks)."""
        obs, info = self.env.reset(seed=seed)
        self._t = 0
        self._last_coverage = float(info.get("coverage", 0.0))
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply a safety-clipped action; pass through the env's reward/terminated."""
        action = np.clip(np.asarray(action, dtype=np.float32), ACTION_LOW, ACTION_HIGH)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        self._last_coverage = float(info.get("coverage", self._last_coverage))
        return obs, float(reward), bool(terminated), bool(truncated), info

    def frame(self) -> np.ndarray:
        """RGB uint8 frame (requires config.render=True)."""
        if not self.config.render:
            raise RuntimeError("PushTInterface created with render=False; set EnvConfig.render=True")
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    # -- read-outs -----------------------------------------------------------
    @property
    def goal_pose(self) -> np.ndarray:
        return np.asarray(self.env.unwrapped.goal_pose, dtype=np.float64)

    @property
    def coverage(self) -> float:
        return self._last_coverage

    @property
    def t(self) -> int:
        return self._t

    @property
    def success(self) -> bool:
        return self._last_coverage > self.success_threshold
