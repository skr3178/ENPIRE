"""Randomized initial-state handling for Push-T.

Two ways to start an episode:

* `episode_seeds(n)` -> deterministic per-episode seeds. `env.reset(seed=s)` then
  produces a randomized *and valid* initial state (the env avoids block/agent
  overlap internally). This is what the rollout harness uses for the 50-episode
  eval -- reproducible and faithful to "50 continuous episodes".

* `compensated_reset_state(obs)` -> a 5-vector to feed to
  `env.reset(options={"reset_to_state": ...})` so that, after the env's legacy
  position-then-angle assignment quirk, the block lands *exactly* at `obs[2:4]`.
  Used by sim-in-the-loop policies (e.g. CEMPolicy) that must sync a private sim
  env to an observation. Derivation recovered from the ENPIRE agent code.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

# Centre-of-gravity offset of the T body in its local frame (recovered constant).
_COG_Y = 45.0


def episode_seeds(n: int, base: int = 0) -> List[int]:
    """Deterministic list of seeds [base, base+1, ..., base+n-1]."""
    return list(range(base, base + n))


def compensated_reset_state(obs: np.ndarray) -> list:
    """Invert the env's position-then-angle CoG quirk.

    After reset the block starts at angle=0 with CoG offset (0, 45). `_set_state`
    sets position then angle; setting angle preserves CoG_world, shifting
    body.position to ``p_req + (45 sin th, 45 (1 - cos th))``. To land the block
    at observed ``b = obs[2:4]`` we request ``p = b - (45 sin th, 45(1-cos th))``.
    """
    ax, ay, bx, by, theta = (float(v) for v in obs)
    px = bx - _COG_Y * math.sin(theta)
    py = by - _COG_Y * (1.0 - math.cos(theta))
    return [ax, ay, px, py, theta]
