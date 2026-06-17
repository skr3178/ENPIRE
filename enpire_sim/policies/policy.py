"""policy.py -- THE FILE THE CODING AGENT EDITS (the PI target).

This is the ENPIRE analog of autoresearch's ``train.py``: the single editable
artifact. Stage 1's coding agent rewrites the body of ``Policy.__call__`` to climb
the Push-T success rate, testing each change against the (immutable) env + rollout
harness. It starts as a deliberately WEAK baseline so the improvement is the agent's
own work -- the recovered Codex/Claude/Kimi policies are NOT given to the agent.

Protocol (do not change the signature):
    Policy(obs) -> np.ndarray  # a 2-vector target position in [0, 512]^2
    Policy.reset()             # called once at the start of each episode

Observation layout (obs_type="state"): [agent_x, agent_y, block_x, block_y, theta].
Goal: push the T-block onto the goal pose at (256, 256), angle pi/4.
"""
from __future__ import annotations

import math

import numpy as np

GOAL_POS = np.array([256.0, 256.0])
GOAL_ANGLE = math.pi / 4


class Policy:
    """Weak baseline: get behind the block, then shove it toward the goal center.

    Two reactive phases: (1) if the agent isn't behind the block (on the far side
    from the goal), move to that staging point; (2) once behind, push through the
    block toward the goal. It ignores the block's ORIENTATION entirely, so it earns
    partial coverage but rarely reaches the 95% threshold -- plenty of headroom for
    the agent to improve (angle alignment, contact choice, settling, etc.).
    """

    def __init__(self, **kwargs):
        pass

    def reset(self) -> None:
        pass

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        agent = obs[:2]
        block = obs[2:4]

        to_goal = GOAL_POS - block
        dist = np.linalg.norm(to_goal)
        direction = to_goal / dist if dist > 1e-6 else np.array([1.0, 0.0])

        behind = block - direction * 35.0  # staging point on the far side from goal
        if np.linalg.norm(agent - behind) > 15.0:
            target = behind                 # phase 1: get into pushing position
        else:
            target = block + direction * 40.0  # phase 2: push through toward goal
        return np.clip(target, 0.0, 512.0).astype(np.float32)
