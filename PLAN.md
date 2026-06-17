# Plan: Reproduce the ENPIRE pipeline (EN→PI→R→E) in simulation, no real robots

## Context

ENPIRE (NVIDIA/CMU/Berkeley, 2026) is **not a model — it's an agent harness**: a coding
agent autonomously improves a robot policy through a closed physical-feedback loop with
four modules (see `ENPIRE.png`):

- **EN — Environment**: construct reset, safety, verification, and logging interfaces the agent can call.
- **PI — Policy Improvement**: generate/revise policy code from rewards, videos, traces, failures.
- **R — Rollout**: run budgeted trials, preserve state/action/video/result for audit.
- **E — Evolution**: N agents on N stations compare branches via Git, reuse winning recipes, prune losers.

**Why "no robots" is fine:** the pipeline is environment-agnostic — a "robot" is anything
exposing `reset()/step()/observation/reward`. gym-pusht already does, and the paper itself
validates in sim (RoboCasa, gym-pusht). The fleet gets *easier*: **N parallel sim instances =
N robot stations** on one machine, and "each station owns its own compute + coding agent,
collaborating via Git" maps almost exactly to **N Claude subagents in N git worktrees**.

**What the env does (gym-pusht):** push a T-shaped block to a goal pose. Verified facts
(`gym-pusht/gym_pusht/envs/pusht.py`): id `gym_pusht/PushT-v0`; `obs_type="state"` →
`[agent_xy, block_xy, block_angle]` float64; action = `Box(0,512,(2,))` target position
(PD-tracked); `reward = clip(coverage/0.95,0,1)`; `terminated` when coverage > 0.95;
`max_episode_steps=300`; `reset(options={"reset_to_state": <5-vec>})` supports fixed inits
but has a **CoM position-then-angle quirk** (the recovered `compensated_reset_state` inverts it);
`render()`→ RGB uint8 frames for video.

**Decisions locked with user:** Full pipeline EN+PI+R+E on Push-T (heuristic). Backend =
**Claude Code subagents** (Anthropic subscription is sufficient; Codex API optional, only for
the cross-agent comparison). One GPU available → enables an optional perception/RL extension.

**Assets already in repo:** `gym-pusht/` cloned; `pusht_components/agent_code/recovered_policies_ordered.py`
holds 3 near-runnable reference policies (`CEMPolicy`, `BeamSearchPushTPolicy`, Kimi replay) — each
needs only `"gym_pusht/PushT-v0"` filled into one `gym.make` plus light stitching;
`pusht_components/pusht_rollouts.json` = reference trajectories; `pusht_components/README.md` = verbatim task prompt.

## Target layout: `enpire_sim/` package

```
enpire_sim/
  envs/
    pusht_interface.py   # EN: auto_reset / safety-clip / verify / log wrapper over gym_pusht/PushT-v0
    init_sampler.py      # randomized initial-state sampler (5-vec, uses compensated_reset_state)
    verify.py            # success = coverage>0.95 over N eps; optional SAM-vision reward (GPU)
  rollout/
    runner.py            # R: run N episodes/policy across parallel workers; record per-episode result
    recorder.py          # save _success/_failure videos, trajectories, reward/coverage logs
  policies/
    baseline_cem.py      # stitched CEMPolicy (strict public-API policy) — Stage-0 validator
    baseline_beam.py     # stitched BeamSearchPushTPolicy
    policy.py            # THE SLOT the coding agent edits (PI)
  agents/
    coding_agent.py      # PI driver: hand task prompt + env API + eval harness to a Claude subagent, loop
    fleet.py             # E: spawn N agents in N git worktrees, shared leaderboard
    evolution.py         # cherry-pick/merge top recipes, prune by avg success
  metrics/
    mru_mtu.py           # sim-MRU, MTU (tokens/min), Tokens-to-Success, Time-to-Success
    plots.py             # hillclimb timeline, 1/4/8 scaling curve, pass@8 table
  configs/task_pusht.yaml  # verbatim prompt, success=100%/50 eps, budgets
  run_single.py          # Stage 1 entry
  run_fleet.py           # Stage 2 entry
  reports/               # generated plots/tables/logs
```

## Build stages

### Stage 0 — Foundation: env interface + baseline + eval (EN + R core)
- `pip install -e gym-pusht`; confirm pymunk/pygame/shapely/opencv import.
- `envs/pusht_interface.py` (**EN**): wrap `gym_pusht/PushT-v0`; expose `auto_reset(seed)` (samples init
  via `init_sampler` + `compensated_reset_state`), `step` with action clipped to [0,512], 300-step budget,
  `verify()` (coverage>0.95), `log()` (frames). Treat raw env code as **immutable** (paper's "no cheating").
- Stitch recovered `CEMPolicy` → `policies/baseline_cem.py` (fill `gym.make("gym_pusht/PushT-v0")`).
- `rollout/runner.py` + `recorder.py` (**R**): run 50 episodes in parallel workers; write
  `reports/results.json` + `_success/_failure` MP4s per episode.
- **Exit:** baseline policy reports a success rate over 50 episodes with videos. Sanity-checks the whole loop.

### Stage 1 — Single-agent autoresearch loop (EN + PI + R closed)
- `agents/coding_agent.py` (**PI**): driver hands a Claude subagent (Agent tool) the verbatim task prompt,
  the `pusht_interface` API doc, and `rollout/runner.py`; loop = agent edits `policies/policy.py` → run
  rollout → feed back success rate + failure videos/metrics → repeat until 100%/50 eps or budget exhausted.
- **Exit:** agent autonomously climbs to ≥99% pass; `reports/iteration_log.jsonl` shows the hill-climb.

### Stage 2 — Multi-agent fleet + Evolution (E)
- `agents/fleet.py`: spawn N coding agents, **each in its own git worktree** off a shared baseline branch
  (worktree ≙ station, own codebase + agent). Each seeded with a distinct hypothesis (approach/hparams).
- `agents/evolution.py`: shared `leaderboard.json` (avg success); agents async **cherry-pick/merge** top
  peer recipes and **prune** losers — the paper's decentralized Git protocol.
- `metrics/mru_mtu.py`: **sim-MRU** = fraction of wall-clock the sim is actively stepping vs agent
  thinking/coding; **MTU** = tokens/min; plus Tokens-to-Success, Time-to-Success.
- Run fleet at **N = 1, 4, 8**.
- **Exit:** worktrees + merges occur autonomously; metrics logged per fleet size.

### Stage 3 — Reproduction artifacts
- `metrics/plots.py`: **hillclimb timeline** (success vs wall-clock, annotated with each agent edit —
  reproduces `ENPIRE.png` bottom), **scaling curve** (time-to-target vs N — reproduces Fig. 7), **pass@8 table**.
- Optional: add Codex API as a second backend → reproduce the Fig. 3 cross-agent comparison.

### Stage 4 — Optional "physical-AI flavor" (uses the GPU)
- `verify.py` vision mode: render frames → SAM/segmentation → coverage-from-pixels reward (mirrors the
  real-robot perception reward + `image.png`).
- Harder task (RoboCasa or robosuite pin-insertion) with BC→offline→online RL in a SERL-style
  deployment/learner/actor split — the conceptual bridge to the physical version. Largest effort; defer.

## Reuse (do not rewrite)
- `pusht_components/agent_code/recovered_policies_ordered.py` — `CEMPolicy`, `BeamSearchPushTPolicy`,
  `compensated_reset_state`, contact heuristics, `_bias_trajectory`. Stitch, don't re-derive.
- `pusht_components/README.md` task prompt → `configs/task_pusht.yaml` verbatim.
- `pusht_components/pusht_rollouts.json` → reference trajectories for plots/validation.
- gym-pusht's own `reset_to_state`, coverage reward, `render()` — used as-is (immutable env).

## Verification (end-to-end)
- **Stage 0:** `python -m enpire_sim.run_single --policy baseline_cem --eval-only --episodes 50`
  → prints success rate, writes `reports/results.json` + videos. Baseline CEM should solve a clear majority.
- **Stage 1:** `python -m enpire_sim.run_single --agent claude --task configs/task_pusht.yaml`
  → `reports/iteration_log.jsonl` shows success climbing across iterations to ≥99%.
- **Stage 2:** `python -m enpire_sim.run_fleet --agents 4 --task configs/task_pusht.yaml`
  → 4 git worktrees created, `leaderboard.json` updates, recipes merge autonomously.
- **Cross-check vs paper:** ~95% success within ~2h wall-clock for a single strong agent; pass@8 ≈ 99%;
  time-to-target drops with fleet size (1→4→8).
