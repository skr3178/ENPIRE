"""E module: evolution / cross-agent recipe sharing over the fleet.

The paper's decentralized protocol: agents "spontaneously cherry-pick, copy, or merge
successful training recipes from their peers." Since a recipe here is a single file
(policy.py), sharing = copy the best station's policy.py into another station's branch
and commit it there, so the laggard continues from a stronger starting point. Merge
events are logged so the idea-tree can draw cross-agent "inspiration" curves.

    python -m enpire_sim.agents.evolution leaderboard
    python -m enpire_sim.agents.evolution share --to 2        # global best -> station w2
    python -m enpire_sim.agents.evolution promote             # global best -> main
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

from enpire_sim.agents.fleet import load as load_fleet, REPO, Station

POLICY_REL = os.path.join("enpire_sim", "policies", "policy.py")
MERGES = os.path.join(REPO, "enpire_sim", "reports", "merges.jsonl")


@dataclass
class LaneBest:
    idx: int
    branch: str
    path: str
    best_sr: float
    best_desc: str


def _git(*args: str, cwd: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _station_log(s: Station) -> str:
    return os.path.join(s.path, "enpire_sim", "reports", "iteration_log.jsonl")


def _best_of(s: Station) -> LaneBest:
    best_sr, best_desc = 0.0, "baseline"
    log = _station_log(s)
    if os.path.exists(log):
        for line in open(log):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if float(r.get("success_rate", 0.0)) > best_sr:
                best_sr = float(r["success_rate"])
                best_desc = r.get("description", "")
    return LaneBest(s.idx, s.branch, s.path, best_sr, best_desc)


def leaderboard() -> List[LaneBest]:
    stations = load_fleet()
    board = sorted((_best_of(s) for s in stations), key=lambda b: b.best_sr, reverse=True)
    return board


def _log_merge(src: LaneBest, dst_branch: str) -> None:
    os.makedirs(os.path.dirname(MERGES), exist_ok=True)
    with open(MERGES, "a") as f:
        f.write(json.dumps({"ts": time.time(), "from": src.branch,
                            "to": dst_branch, "sr": src.best_sr}) + "\n")


def share(to_idx: int) -> None:
    """Copy the global-best station's policy.py into station `to_idx` and commit there."""
    board = leaderboard()
    if not board:
        print("no stations; run fleet setup first")
        return
    best = board[0]
    stations = {s.idx: s for s in load_fleet()}
    if to_idx not in stations:
        print(f"no station w{to_idx}")
        return
    dst = stations[to_idx]
    if dst.branch == best.branch:
        print(f"station w{to_idx} is already the best ({best.best_sr*100:.0f}%); nothing to share")
        return
    shutil.copyfile(os.path.join(best.path, POLICY_REL), os.path.join(dst.path, POLICY_REL))
    _git("add", POLICY_REL, cwd=dst.path)
    _git("-c", "user.name=enpire-evolution", "-c", "user.email=evo@enpire",
         "commit", "-m", f"merge: adopt recipe from {best.branch} ({best.best_sr*100:.0f}%)", cwd=dst.path)
    _log_merge(best, dst.branch)
    print(f"shared {best.branch} ({best.best_sr*100:.0f}%) -> w{to_idx} [{dst.branch}]")


def promote() -> None:
    """Copy the global-best policy.py onto main and commit."""
    board = leaderboard()
    if not board or board[0].best_sr <= 0:
        print("no successful policy to promote yet")
        return
    best = board[0]
    shutil.copyfile(os.path.join(best.path, POLICY_REL), os.path.join(REPO, POLICY_REL))
    _git("add", POLICY_REL, cwd=REPO)
    _git("-c", "user.name=enpire-evolution", "-c", "user.email=evo@enpire",
         "commit", "-m", f"promote best policy from {best.branch} ({best.best_sr*100:.0f}%)", cwd=REPO)
    print(f"promoted {best.branch} ({best.best_sr*100:.0f}%) -> main (review, then push)")


def main() -> None:
    p = argparse.ArgumentParser(description="ENPIRE evolution (cross-agent recipe sharing).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("leaderboard")
    sh = sub.add_parser("share"); sh.add_argument("--to", type=int, required=True)
    sub.add_parser("promote")
    args = p.parse_args()

    if args.cmd == "leaderboard":
        for b in leaderboard():
            print(f"  w{b.idx}  {b.best_sr*100:5.1f}%  [{b.branch}]  best idea: {b.best_desc}")
    elif args.cmd == "share":
        share(args.to)
    elif args.cmd == "promote":
        promote()


if __name__ == "__main__":
    main()
