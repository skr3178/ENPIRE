"""Verification (the EN module's 'verify' interface).

Ground-truth success for Push-T is the env's own coverage signal: the episode
succeeds when goal-coverage exceeds the env threshold (0.95). We treat the env's
reward/terminated as immutable (the paper's "no cheating" rule) and only read it.

`verify.py` is also where an optional perception-based reward would live
(Stage 4): render frames -> segment the T -> compute coverage from pixels. That
mirrors the real-robot reward path and is gated behind a GPU; not implemented yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

# Documented mirror of gym-pusht's PushTEnv.success_threshold. Callers should
# prefer the live env value (PushTInterface.success_threshold); this is only a
# fallback default for when no env is on hand.
SUCCESS_COVERAGE = 0.95


def episode_success(final_coverage: float, threshold: float = SUCCESS_COVERAGE) -> bool:
    """An episode is a success iff final coverage exceeds the threshold."""
    return float(final_coverage) > threshold


@dataclass
class VerifySummary:
    n: int
    n_success: int
    success_rate: float
    mean_coverage: float
    mean_steps: float


def summarize(coverages: List[float], successes: List[bool], steps: List[int]) -> VerifySummary:
    """Aggregate per-episode results into the verification summary the agent reads."""
    n = len(successes)
    n_success = int(sum(bool(s) for s in successes))
    return VerifySummary(
        n=n,
        n_success=n_success,
        success_rate=(n_success / n) if n else 0.0,
        mean_coverage=(sum(coverages) / n) if n else 0.0,
        mean_steps=(sum(steps) / n) if n else 0.0,
    )
