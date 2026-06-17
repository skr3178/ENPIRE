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
        self.K_ANG = 100.0
        self.R = 55.0

    def reset(self) -> None:
        pass

    def _cost(self, block, theta):
        pe = np.linalg.norm(GOAL_POS - block)
        ae = abs(((theta - GOAL_ANGLE + math.pi) % (2 * math.pi)) - math.pi)
        return pe + self.K_ANG * ae

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        agent = obs[:2]
        block = obs[2:4]
        theta = obs[4]

        base = self._cost(block, theta)
        best_c = None
        best_d = None
        best_score = -1e18
        for k in range(16):
            ang = 2 * math.pi * k / 16
            c = block + self.R * np.array([math.cos(ang), math.sin(ang)])
            d = block - c
            d = d / (np.linalg.norm(d) + 1e-9)
            nb = block + d * 10.0
            lever = c - block
            torque = lever[0] * d[1] - lever[1] * d[0]
            nth = theta + 0.0008 * torque
            score = base - self._cost(nb, nth)
            if score > best_score:
                best_score = score
                best_c, best_d = c, d

        # two-phase: stage behind contact, then push through center
        if np.linalg.norm(agent - best_c) > 12.0:
            target = best_c
        else:
            target = block + best_d * 40.0
        return np.clip(target, 0.0, 512.0).astype(np.float32)
