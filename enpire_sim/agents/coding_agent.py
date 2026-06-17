"""PI module: the coding-agent iteration helper.

The coding agent (a Claude subagent) runs the autoresearch loop on a dedicated git
branch. Each iteration it edits policy.py, commits, then runs ONE evaluation through
this helper, which:

  * runs the rollout eval (the immutable R harness),
  * appends a row to reports/iteration_log.jsonl  (machine-readable hill-climb),
  * appends a row to reports/results.tsv          (autoresearch-style log),
  * prints success_rate + whether it beat the running best on this branch.

The agent then decides keep (leave the commit) or revert (git reset) from the print.

Usage (one iteration):
    python -m enpire_sim.agents.coding_agent eval --episodes 20 --desc "angle align first"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from enpire_sim.envs.init_sampler import episode_seeds
from enpire_sim.rollout.runner import RolloutConfig, evaluate

LOG_JSONL = "enpire_sim/reports/iteration_log.jsonl"
LOG_TSV = "enpire_sim/reports/results.tsv"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return ""


def git_head() -> str:
    return _git("rev-parse", "--short", "HEAD") or "nogit"


def git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD") or "nobranch"


def branch_best(branch: str) -> float:
    """Best success_rate logged so far on this branch (for keep/revert comparison)."""
    if not os.path.exists(LOG_JSONL):
        return 0.0
    best = 0.0
    with open(LOG_JSONL) as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("branch") == branch:
                best = max(best, float(row.get("success_rate", 0.0)))
    return best


def _append_jsonl(row: dict) -> None:
    os.makedirs(os.path.dirname(LOG_JSONL), exist_ok=True)
    with open(LOG_JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def _append_tsv(row: dict) -> None:
    os.makedirs(os.path.dirname(LOG_TSV), exist_ok=True)
    new = not os.path.exists(LOG_TSV)
    with open(LOG_TSV, "a") as f:
        if new:
            f.write("commit\tbranch\tsuccess_rate\tmean_coverage\tepisodes\tdescription\n")
        f.write(f"{row['commit']}\t{row['branch']}\t{row['success_rate']:.3f}\t"
                f"{row['mean_coverage']:.3f}\t{row['episodes']}\t{row['description']}\n")


def cmd_eval(args: argparse.Namespace) -> None:
    seeds = episode_seeds(args.episodes, base=args.base_seed)
    config = RolloutConfig(max_steps=args.max_steps,
                           out_dir="enpire_sim/reports/rollouts",
                           video_prefix="policy")
    branch, prev_best = git_branch(), 0.0
    prev_best = branch_best(branch)

    t0 = time.perf_counter()
    summary = evaluate(("policy", {}), seeds, config=config, workers=args.workers)
    elapsed = time.perf_counter() - t0

    row = {
        "ts": time.time(),
        "branch": branch,
        "commit": git_head(),
        "success_rate": summary["success_rate"],
        "mean_coverage": summary["mean_coverage"],
        "mean_steps": summary["mean_steps"],
        "episodes": summary["n_episodes"],
        "eval_seconds": elapsed,
        "description": args.desc,
    }
    _append_jsonl(row)
    _append_tsv(row)

    sr = summary["success_rate"]
    improved = sr > prev_best
    print(f"success_rate={sr*100:.1f}%  mean_cov={summary['mean_coverage']:.3f}  "
          f"({summary['n_success']}/{summary['n_episodes']})  eval={elapsed:.0f}s")
    print(f"prev_best_on_branch={prev_best*100:.1f}%  -> {'IMPROVED (keep)' if improved else 'NOT IMPROVED (revert)'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Coding-agent iteration helper (PI).")
    sub = p.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval", help="run one eval + log it as an iteration")
    ev.add_argument("--episodes", type=int, default=20)
    ev.add_argument("--base-seed", type=int, default=0)
    ev.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    ev.add_argument("--max-steps", type=int, default=300)
    ev.add_argument("--desc", default="", help="one-line description of this experiment")
    ev.set_defaults(func=cmd_eval)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
