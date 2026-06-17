"""E module: the multi-agent fleet over git worktrees.

Each "robot station" is a git worktree: a separate working directory bound to its own
branch, all sharing the one .git. N coding agents edit their own policy.py on their own
branch in parallel, never colliding -- the faithful realization of "each station owns
its compute + agent, collaborating via Git."

Worktrees are created as siblings of the main repo (kept out of it):
    <repo>/../ENPIRE-fleet/w0   on branch autoresearch/run-w0
    <repo>/../ENPIRE-fleet/w1   on branch autoresearch/run-w1
    ...
Each reuses the main repo's .venv + gym-pusht install (agents call the main
interpreter with their worktree as CWD), so there is no per-station reinstall.

    python -m enpire_sim.agents.fleet setup --n 4
    python -m enpire_sim.agents.fleet list
    python -m enpire_sim.agents.fleet teardown
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from typing import List

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLEET_DIR = os.path.normpath(os.path.join(REPO, "..", "ENPIRE-fleet"))
BASE_BRANCH = "enpire/baseline"
BRANCH_PREFIX = "autoresearch/run-w"
MANIFEST = os.path.join(REPO, "enpire_sim", "reports", "fleet.json")


@dataclass
class Station:
    idx: int
    path: str       # worktree directory
    branch: str     # autoresearch/run-wN
    venv_python: str


def _git(*args: str, cwd: str = REPO) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _existing_branches() -> set:
    out = _git("branch", "--format=%(refname:short)")
    return set(out.splitlines())


def setup(n: int, base: str = BASE_BRANCH) -> List[Station]:
    """Create n worktrees (each a fresh branch off `base`)."""
    os.makedirs(FLEET_DIR, exist_ok=True)
    venv_python = os.path.join(REPO, ".venv", "bin", "python")
    branches = _existing_branches()
    stations: List[Station] = []
    for i in range(n):
        path = os.path.join(FLEET_DIR, f"w{i}")
        branch = f"{BRANCH_PREFIX}{i}"
        if os.path.exists(path):
            print(f"  w{i}: worktree already exists at {path} (skipping)")
        else:
            args = ["worktree", "add"]
            if branch in branches:
                args += [path, branch]                 # reuse existing branch
            else:
                args += [path, "-b", branch, base]     # new branch off base
            _git(*args)
            print(f"  w{i}: {path}  [{branch}]")
        stations.append(Station(idx=i, path=path, branch=branch, venv_python=venv_python))

    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump([asdict(s) for s in stations], f, indent=2)
    print(f"wrote manifest {MANIFEST}")
    return stations


def load() -> List[Station]:
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST) as f:
        return [Station(**d) for d in json.load(f)]


def teardown(remove_branches: bool = False) -> None:
    for s in load():
        if os.path.exists(s.path):
            try:
                _git("worktree", "remove", "--force", s.path)
                print(f"  removed {s.path}")
            except subprocess.CalledProcessError as e:
                print(f"  could not remove {s.path}: {e}")
    _git("worktree", "prune")
    if remove_branches:
        for s in load():
            try:
                _git("branch", "-D", s.branch)
            except subprocess.CalledProcessError:
                pass
    if os.path.exists(MANIFEST):
        os.remove(MANIFEST)


def main() -> None:
    p = argparse.ArgumentParser(description="ENPIRE fleet (git-worktree stations).")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("--n", type=int, default=4)
    s.add_argument("--base", default=BASE_BRANCH)
    sub.add_parser("list")
    td = sub.add_parser("teardown"); td.add_argument("--branches", action="store_true")
    args = p.parse_args()

    if args.cmd == "setup":
        setup(args.n, args.base)
    elif args.cmd == "list":
        for s in load():
            exists = "ok" if os.path.exists(s.path) else "MISSING"
            print(f"  w{s.idx}  [{s.branch}]  {s.path}  ({exists})")
    elif args.cmd == "teardown":
        teardown(remove_branches=args.branches)


if __name__ == "__main__":
    main()
