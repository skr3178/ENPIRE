"""Stage 2 fleet orchestration entry.

This wires the NON-agent parts of a fleet run into one place: create the worktrees,
codify each station's starting hypothesis, report status, and render Figure 1. The
coding agents themselves are launched by the orchestrator (the Claude Code session)
spawning one subagent per station from the launch spec written by `setup` -- see
`enpire_sim/configs/task_pusht.md` for the per-agent protocol.

    python -m enpire_sim.run_fleet setup --n 4     # create worktrees + write launch spec
    python -m enpire_sim.run_fleet status          # leaderboard + iteration counts
    python -m enpire_sim.run_fleet render           # write Figure 1 (idea tree + hillclimb)
    python -m enpire_sim.run_fleet teardown

Then the orchestrator launches one coding subagent per station (reports/fleet_launch.json),
each working in its worktree, editing policy.py via the autoresearch loop.
"""
from __future__ import annotations

import argparse
import json
import os

from enpire_sim.agents import fleet as fleet_mod
from enpire_sim.agents import evolution as evo

# The canonical fleet plan: each station's distinct starting hypothesis (its lane).
# Extend/reorder to scale to 1, 4, or 8 stations (paper's scaling axis).
HYPOTHESES = [
    ("two-phase reactive", "Phase ROTATE: push tangentially on a bar-tip to spin the T toward goal angle. "
                           "Phase TRANSLATE: when aligned, push through the centre of mass toward (256,256)."),
    ("greedy contact selection", "Each step, enumerate candidate contacts around the T + a push direction; "
                                 "score by reduction in pos_error + k*angle_error; execute the best (1-step greedy)."),
    ("short internal-sim lookahead", "Private sim synced via compensated_reset_state; try <=8 candidate targets over "
                                     "<=3 step rollouts; pick lowest block-pose cost. Keep it fast."),
    ("angle-priority PD", "Smooth proportional law gated on orientation: correct angle first (offset torque push), "
                          "then drive position proportionally to goal. Tune gains + switch threshold."),
    ("potential field", "Attractive field to the goal pose + repulsive shaping so contacts approach from behind the block."),
    ("keypoint alignment", "Align the T's keypoints to the goal-pose keypoints one at a time (rotation then translation)."),
    ("bang-bang switching", "Discrete push primitives toward target sub-goals with hysteresis to avoid chatter."),
    ("coarse-to-fine MPC", "Large pushes far from goal, small corrective pushes near it; widen/narrow candidate set by distance."),
]
LAUNCH_SPEC = "enpire_sim/reports/fleet_launch.json"


def setup(n: int) -> None:
    stations = fleet_mod.setup(n)
    spec = []
    for s in stations:
        name, hint = HYPOTHESES[s.idx % len(HYPOTHESES)]
        spec.append({"idx": s.idx, "path": s.path, "branch": s.branch,
                     "venv_python": s.venv_python, "hypothesis": name, "hint": hint})
    os.makedirs(os.path.dirname(LAUNCH_SPEC), exist_ok=True)
    with open(LAUNCH_SPEC, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"\nwrote launch spec {LAUNCH_SPEC} for {n} stations:")
    for st in spec:
        print(f"  w{st['idx']}  [{st['branch']}]  -> {st['hypothesis']}")
    print("\nNext: the orchestrator launches one coding subagent per station (each in its worktree).")


def status() -> None:
    board = evo.leaderboard()
    if not board:
        print("no fleet set up yet (run: python -m enpire_sim.run_fleet setup --n 4)")
        return
    print("station        best     iterations  best idea")
    stations = {s.idx: s for s in fleet_mod.load()}
    for b in board:
        log = os.path.join(stations[b.idx].path, "enpire_sim", "reports", "iteration_log.jsonl")
        nit = sum(1 for _ in open(log)) if os.path.exists(log) else 0
        print(f"  w{b.idx} [{b.branch.split('/')[-1]}]   {b.best_sr*100:5.1f}%   {nit:>3}        {b.best_desc}")


def render(out: str) -> None:
    from enpire_sim.metrics import idea_tree
    logs = idea_tree.default_logs()
    idea_tree.figure1(idea_tree.collect(logs), out)


def main() -> None:
    p = argparse.ArgumentParser(description="ENPIRE fleet orchestration (Stage 2).")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("--n", type=int, default=4)
    sub.add_parser("status")
    r = sub.add_parser("render"); r.add_argument("--out", default="enpire_sim/reports/figure1.png")
    td = sub.add_parser("teardown"); td.add_argument("--branches", action="store_true")
    args = p.parse_args()

    if args.cmd == "setup":
        setup(args.n)
    elif args.cmd == "status":
        status()
    elif args.cmd == "render":
        render(args.out)
    elif args.cmd == "teardown":
        fleet_mod.teardown(remove_branches=args.branches)


if __name__ == "__main__":
    main()
