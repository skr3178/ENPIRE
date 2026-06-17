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
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        agent = obs[:2]
        block = obs[2:4]
        theta = float(obs[4])
        self._step += 1

        to_goal = GOAL_POS - block
        dist = float(np.linalg.norm(to_goal))
        dn = to_goal / dist if dist > 1e-6 else np.array([1.0, 0.0])
        ang_err = math.atan2(math.sin(GOAL_ANGLE - theta), math.cos(GOAL_ANGLE - theta))

        POS_TOL = 30.0
        if dist > POS_TOL:
            # TRANSLATE: get behind block (far side from goal), push through toward goal
            behind = block - dn * 45.0
            if np.linalg.norm(agent - behind) > 16.0:
                target = behind
            else:
                target = block + dn * 18.0
        else:
            # ROTATE in place: tangential contact to spin block toward goal angle
            radial = np.array([math.cos(theta), math.sin(theta)])
            tang = np.array([-radial[1], radial[0]])
            s = 1.0 if ang_err > 0 else -1.0
            contact = block + radial * 38.0 * s
            if np.linalg.norm(agent - contact) > 16.0:
                target = contact
            else:
                target = contact - tang * s * 18.0

        step = target - agent
        sl = float(np.linalg.norm(step))
        if sl > 18.0:
            target = agent + step / sl * 18.0
        return np.clip(target, 0.0, 512.0).astype(np.float32)
