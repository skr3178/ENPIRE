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
POLICY = "enpire_sim/policies/policy.py"
ADOPT_MARGIN = 0.10  # adopt a peer only if it beats your branch-best by >= this


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return ""


def shared_dir() -> str:
    """The cross-worktree shared state dir (ENPIRE-fleet/shared), resolved from the
    common .git so it is identical from any worktree or the main repo."""
    common = _git("rev-parse", "--git-common-dir")          # <main-repo>/.git
    main_repo = os.path.dirname(os.path.abspath(common)) if common else os.getcwd()
    d = os.path.normpath(os.path.join(main_repo, "..", "ENPIRE-fleet", "shared"))
    os.makedirs(d, exist_ok=True)
    return d


def _shared(path: str) -> str:
    return os.path.join(shared_dir(), path)


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
    # Publish this station's best to the shared leaderboard so peers can see it (E module).
    best_so_far = max(prev_best, summary["success_rate"])
    with open(_shared("leaderboard.jsonl"), "a") as f:
        f.write(json.dumps({"ts": time.time(), "branch": branch, "commit": git_head(),
                            "success_rate": best_so_far,
                            "mean_coverage": summary["mean_coverage"]}) + "\n")

    sr = summary["success_rate"]
    improved = sr > prev_best
    print(f"success_rate={sr*100:.1f}%  mean_cov={summary['mean_coverage']:.3f}  "
          f"({summary['n_success']}/{summary['n_episodes']})  eval={elapsed:.0f}s")
    print(f"prev_best_on_branch={prev_best*100:.1f}%  -> {'IMPROVED (keep)' if improved else 'NOT IMPROVED (revert)'}")


def _leaderboard_latest() -> dict:
    """Latest best-per-branch from the shared leaderboard: {branch: (sr, commit, cov)}."""
    path = _shared("leaderboard.jsonl")
    out: dict = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            b = r.get("branch")
            sr = float(r.get("success_rate", 0.0))
            if b and (b not in out or sr >= out[b][0]):
                out[b] = (sr, r.get("commit", "?"), float(r.get("mean_coverage", 0.0)))
    return out


def cmd_peek(args: argparse.Namespace) -> None:
    """Show peers' best vs your own; flag any peer worth adopting (E module)."""
    me = git_branch()
    board = _leaderboard_latest()
    my_best = board.get(me, (branch_best(me), "?", 0.0))[0]
    print(f"you are [{me}], branch-best success={my_best*100:.1f}%")
    print("peers (shared leaderboard):")
    peers = sorted(((b, v) for b, v in board.items() if b != me), key=lambda x: -x[1][0])
    if not peers:
        print("  (no peers have published yet)")
    for b, (sr, commit, cov) in peers:
        flag = "  <-- ADOPT-WORTHY (>=10pp ahead)" if sr >= my_best + ADOPT_MARGIN else ""
        print(f"  {b:<26} best={sr*100:5.1f}%  cov={cov:.2f}  ({commit}){flag}")
    print(f"\nIf a peer is ADOPT-WORTHY and you have stalled, run:\n"
          f"  python -m enpire_sim.agents.coding_agent adopt --from <peer-branch>")


def cmd_adopt(args: argparse.Namespace) -> None:
    """Pull a peer's committed policy.py into this branch and commit it (E module)."""
    src = args.from_branch
    me = git_branch()
    policy_src = _git("show", f"{src}:{POLICY}")
    if not policy_src:
        print(f"could not read {src}:{POLICY} (peer branch/policy not found)")
        return
    with open(POLICY, "w") as f:
        f.write(policy_src + ("\n" if not policy_src.endswith("\n") else ""))
    subprocess.run(["git", "add", POLICY], check=False)
    subprocess.run(["git", "-c", "user.name=enpire-agent", "-c", "user.email=a@enpire",
                    "commit", "-q", "-m", f"merge: adopt recipe from {src}"], check=False)
    board = _leaderboard_latest()
    sr = board.get(src, (0.0,))[0]
    with open(_shared("merges.jsonl"), "a") as f:
        f.write(json.dumps({"ts": time.time(), "from": src, "to": me, "sr": sr}) + "\n")
    print(f"adopted {src}'s policy ({sr*100:.0f}%) into [{me}] and logged the merge. "
          f"Now eval it, then IMPROVE on it with your lane's own strength.")


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

    pk = sub.add_parser("peek", help="see peers' best on the shared leaderboard (E module)")
    pk.set_defaults(func=cmd_peek)

    ad = sub.add_parser("adopt", help="adopt a peer's policy.py into your branch (E module)")
    ad.add_argument("--from", dest="from_branch", required=True, help="peer branch, e.g. autoresearch/run-w3")
    ad.set_defaults(func=cmd_adopt)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
