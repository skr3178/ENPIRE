# ENPIRE (sim reproduction)

A simulation-only reproduction of the ENPIRE physical-autoresearch pipeline
(EN → PI → R → E) on the **Push-T** task — no real robots. A "robot station" is a
`gym-pusht` sim instance; the fleet is N such instances driven by N coding agents in
N git worktrees.

Paper: *ENPIRE: Agentic Robot Policy Self-Improvement in the Real World* (NVIDIA / CMU /
UC Berkeley, 2026). See [`PLAN.md`](PLAN.md) for the full build plan and stage map.

## Status

- **Stage 0 (EN + R) — done.** Env interface, parallel rollout/eval, video+results
  logging. Recovered CEM baseline validated at **96% success (48/50)** over 50 episodes.
- **Stage 1 (PI)** — single-agent autoresearch loop: an agent edits `policy.py` from a
  weak stub until Push-T is solved. *(in progress)*
- **Stage 2 (E)** — multi-agent fleet over git worktrees + evolution. *(pending)*
- **Stage 3** — reproduction plots (hill-climb, scaling, pass@8). *(pending)*

## Setup

```bash
# 1. The immutable Push-T env (not vendored)
git clone https://github.com/huggingface/gym-pusht
# 2. Python env (uv); pin pymunk < 7 (7.x removed add_collision_handler)
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e gym-pusht 'pymunk>=6.6,<7' imageio-ffmpeg
```

## Run

```bash
# Evaluate the recovered CEM baseline (Stage 0 smoke test)
python -m enpire_sim.run_single --policy cem --episodes 50 --workers 24
# Evaluate the weak editable stub the agent improves
python -m enpire_sim.run_single --policy policy --episodes 50 --videos
```

## Layout

```
enpire_sim/        the reproduction harness (EN / PI / R / E)
  envs/            EN: PushTInterface, init sampler, verification
  rollout/         R: parallel runner + video/trajectory recorder
  policies/        policy.py (agent-edited stub) + baseline_cem (reference)
  configs/         task_pusht.md (the agent's standing protocol)
pusht_components/  assets recovered from the ENPIRE website (recovered agent
                   policies, task prompt, rollout traces). Large media gitignored.
PLAN.md            full plan + stage map
```

Provenance note: `policies/baseline_cem.py` and `pusht_components/agent_code/` are
ENPIRE's own agent-written code recovered from the project website; the harness
(env interface, rollout, fleet) is original. See `PLAN.md`.
