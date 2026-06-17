# Push-T autoresearch task (the coding agent's standing protocol)

You are an autonomous coding agent improving a robot policy for the **Push-T** task.
This protocol is adapted from Karpathy's `autoresearch/program.md`, retargeted from
nanochat training to a Push-T heuristic policy (the ENPIRE PI loop).

## The task

Write a heuristic policy, **with no neural-network training**, to achieve a 100%
success rate in the Push-T environment over at least 50 continuous episodes. You are
**not allowed to modify environment code; that is cheating. No cheating.** For each
policy evaluation, a video is saved with a `_success` or `_failure` suffix.

Success = the env's own signal: block-on-goal coverage > 0.95 (sourced from
`gym_pusht`, immutable).

## What you CAN do
- Edit **only** `enpire_sim/policies/policy.py` — the `Policy` class. Everything is
  fair game: the control law, contact selection, angle alignment, multi-phase plans,
  an internal sim for planning (CEM/beam/etc.), hyperparameters.

## What you CANNOT do
- Modify the env (`gym-pusht/`), the rollout harness (`enpire_sim/envs`,
  `enpire_sim/rollout`), or the verification. These are the ground-truth eval.
- Change the `Policy(obs) -> np.ndarray` / `Policy.reset()` signature.
- Read or copy the recovered reference policies in `pusht_components/agent_code/`.
  You must write the policy yourself.

## Observation / action
- obs (state): `[agent_x, agent_y, block_x, block_y, block_angle]`.
- action: a 2-vector target position in `[0, 512]^2` (the agent is PD-tracked to it).
- Goal pose: block at `(256, 256)`, angle `pi/4`.

## The experiment loop (run on a dedicated git branch/worktree)

LOOP until success rate hits the target or you are stopped:

1. Look at the current git state (branch/commit).
2. Edit `policy.py` with one experimental idea (hack the code directly).
3. `git commit` the change.
4. Run the evaluation:
   `python -m enpire_sim.run_single --policy policy --episodes 50 --workers 32`
5. Read the result from `enpire_sim/reports/results_policy.json`
   (`success_rate`, `mean_coverage`, plus `_failure` videos in `reports/rollouts/`).
6. If a run crashes, read the traceback, fix obvious bugs, else log a crash and move on.
7. **Keep or revert**: if `success_rate` improved, advance the branch (keep the
   commit). If equal or worse, `git reset` back to where you started this iteration.
8. Log the iteration (commit, success_rate, mean_coverage, status, one-line description).

## Guidance
- Inspect `_failure` videos / trajectories to decide what to change next (the block's
  orientation is usually the hard part — the weak baseline ignores it entirely).
- **Simplicity criterion**: all else equal, simpler is better. A small gain that adds
  ugly complexity is not worth it; a simplification with equal results is a win.
- **NEVER STOP** to ask "should I keep going?" — iterate autonomously until the target
  is reached or you are interrupted. If out of ideas, inspect failures harder, try
  angle-alignment before translation, try an internal planning sim.
