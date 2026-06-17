# PushT components pulled from the ENPIRE website

Source: https://research.nvidia.com/labs/gear/enpire (paper: *ENPIRE: Agentic Robot
Policy Self-Improvement in the Real World*, NVIDIA / CMU / UC Berkeley, 2026).

These are the PushT (Push-T) related artifacts extractable from the project page. The
real environment code lives in the upstream repo, referenced below.

## Upstream environment
- `huggingface/gym-pusht` — https://github.com/huggingface/gym-pusht
  The page instructs: *"Clone huggingface/gym-pusht"*. This is the actual Push-T env;
  use it for reproduction. The website does not ship its own env code.

## Task prompt (verbatim from the site)
> Write a heuristic policy, with no neural network training, to achieve a 100% success
> rate in the Push-T environment over at least 50 continuous episodes. You are not
> allowed to modify environment code; that is cheating. No cheating. Fan out a subagent
> team to try approaches. For each policy evaluation, save a video with a unique name
> and a `_success` or `_failure` suffix.

Reported result: **99% pass@8** success rate. Agents that solved it: Codex, Claude Code,
Kimi Code.

## Files here
- `pusht_rollouts.json` — geometry + trajectories for the three agent rollouts, in the
  page's 512×512 SVG coordinate space:
  - `goal_transform` / `goal_paths` — the target **T** pose (`translate(256 256) rotate(45)`)
    and the two rectangles that compose the T (cross-bar + stem).
  - `block_start_transform` — the object's initial pose, identical across agents:
    `translate(240.223 210.674) rotate(79.33...)`.
  - `trace_d` — raw SVG path of the end-effector/object trace.
  - `trace_points` — same trace parsed into `[x, y]` pairs (Codex 44, Claude Code 74,
    Kimi Code 67 points).
- `rollout_{codex,claude_code,kimi_code}.svg` — standalone renderable SVGs of each rollout.
- `env_skeleton.py` — the illustrative `env.py` abstraction shown on the site
  (`reset` / `get_reward` / `get_observation` / `step`). Pseudocode, not runnable.
- `pusht-reset-only-{1..4}.mp4` — the four randomized auto-reset clips for Push-T.
- `pusht-success.mp4` — a successful Push-T rollout clip.

## Coordinate notes (for reproduction)
The visualizations use a 512×512 canvas with a 128px grid. The goal T is centered at
(256, 256) rotated 45°, built from two rects: cross-bar `M -60 30 L 60 30 L 60 0 L -60 0 Z`
and stem `M -15 30 L -15 120 L 15 120 L 15 30 Z`. These are display coordinates, not the
gym-pusht physics units — map them to the real env's frame before use.
