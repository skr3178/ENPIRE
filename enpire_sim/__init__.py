"""ENPIRE-sim: a simulation-only reproduction of the ENPIRE physical-autoresearch
pipeline (EN -> PI -> R -> E) on the gym-pusht Push-T task.

A "robot station" is a sim env instance; the fleet is N such instances driven by
N coding agents in N git worktrees. See PLAN.md at the repo root.
"""
import os

# Run pygame/pymunk headless by default so env stepping + rgb_array rendering work
# on machines without a display (and inside worker processes / subagents).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

__version__ = "0.0.1"
