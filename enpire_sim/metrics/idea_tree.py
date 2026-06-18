"""Stage 3: the ENPIRE "Figure 1" -- idea tree (top) + team hillclimb (bottom).

Reproduces the paper's combined figure: each coding agent explores its own lane of
ideas; every dot is an idea it tried; a green ring marks an idea that raised the
team's best success rate; the lower panel tracks the team-average best success rate
over research wall-clock time. (Cross-agent "inspiration" curves are drawn when the
evolution module logs a merge -- see enpire_sim/agents/evolution.py.)

Data: each station logs to its own worktree at
  <ENPIRE-fleet>/wN/enpire_sim/reports/iteration_log.jsonl
plus the main repo's log. Rows are tagged by `branch` = the lane.

    python -m enpire_sim.metrics.idea_tree --out enpire_sim/reports/figure1.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLEET_DIR = os.path.normpath(os.path.join(REPO, "..", "ENPIRE-fleet"))


def default_logs() -> List[str]:
    logs = glob.glob(os.path.join(FLEET_DIR, "w*", "enpire_sim", "reports", "iteration_log.jsonl"))
    main_log = os.path.join(REPO, "enpire_sim", "reports", "iteration_log.jsonl")
    if os.path.exists(main_log):
        logs.append(main_log)
    return sorted(logs)


def collect(logs: List[str]) -> List[dict]:
    rows = []
    for path in logs:
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    rows = [r for r in rows if "ts" in r and "branch" in r]
    rows.sort(key=lambda r: r["ts"])
    return rows


def load_merges() -> List[dict]:
    """Cross-agent recipe-sharing events. Agent-driven adoptions are logged to the
    shared fleet dir (ENPIRE-fleet/shared/merges.jsonl); the orchestrator-driven
    evolution.share log (REPO/enpire_sim/reports/merges.jsonl) is read as a fallback."""
    paths = [
        os.path.join(FLEET_DIR, "shared", "merges.jsonl"),   # agent-driven (peek/adopt)
        os.path.join(REPO, "enpire_sim", "reports", "merges.jsonl"),  # orchestrator-driven
    ]
    out = []
    for path in paths:
        if os.path.exists(path):
            for line in open(path):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def figure1(rows: List[dict], out: str, time_unit: str = "min") -> Optional[str]:
    if not rows:
        print("no iterations logged yet; nothing to plot")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    div = 60.0 if time_unit == "min" else 3600.0
    t0 = rows[0]["ts"]
    lanes = sorted({r["branch"] for r in rows})
    lane_idx = {b: i for i, b in enumerate(lanes)}

    green, dgreen, grey = "#7e9e6b", "#4f6b3f", "#9aa0a6"

    # Walk rows in time order: per-lane best, team-average best, ring-on-improvement.
    lane_best: Dict[str, float] = {b: 0.0 for b in lanes}
    nodes = []          # (t, lane_i, sr, is_ring)
    team_t, team_avg = [], []
    for r in rows:
        b = r["branch"]
        t = (r["ts"] - t0) / div
        sr = float(r.get("success_rate", 0.0)) * 100.0
        ring = sr > lane_best[b] + 1e-9
        if ring:
            lane_best[b] = sr
        nodes.append((t, lane_idx[b], sr, ring))
        team_t.append(t)
        team_avg.append(sum(lane_best.values()) / len(lanes))

    fig, (ax_tree, ax_hc) = plt.subplots(
        2, 1, figsize=(11, 6), height_ratios=[1.5, 1], sharex=True)

    # -- top: idea tree (lanes) --
    for b, i in lane_idx.items():
        ts = [t for (t, li, _, _) in nodes if li == i]
        if ts:
            ax_tree.plot([min(ts), max(ts)], [i, i], color=grey, lw=0.8, zorder=1)
    for (t, li, sr, ring) in nodes:
        ax_tree.scatter([t], [li], s=18, color=dgreen, zorder=3)
        if ring:
            ax_tree.scatter([t], [li], s=90, facecolors="none",
                            edgecolors=green, linewidths=1.8, zorder=4)

    # cross-agent inspiration curves (adoptions): arrow from the SOURCE idea node that was
    # adopted (the source lane's best node at/before the merge) -> the adoption point on the
    # target lane (at the merge time). This references the actual recipe, not just the lanes.
    import matplotlib.patches as mpatches
    for m in load_merges():
        src, dst = m.get("from"), m.get("to")
        if src not in lane_idx or dst not in lane_idx:
            continue
        merge_x = (m["ts"] - t0) / div
        si, di = lane_idx[src], lane_idx[dst]
        # the adopted recipe = the source lane's highest-success node at/before the merge
        src_nodes = [(t, sr) for (t, li, sr, _) in nodes if li == si and t <= merge_x + 1e-9]
        src_x = max(src_nodes, key=lambda x: (x[1], x[0]))[0] if src_nodes else merge_x
        arc = mpatches.FancyArrowPatch(
            (src_x, si), (merge_x, di),
            connectionstyle="arc3,rad=0.25", color=green, lw=1.5,
            alpha=0.85, arrowstyle="-|>", mutation_scale=13, zorder=2)
        ax_tree.add_patch(arc)
        # mark the source recipe node so the reference is unambiguous
        ax_tree.scatter([src_x], [si], s=70, facecolors="none",
                        edgecolors=green, linewidths=1.2, linestyle=":", zorder=4)
    ax_tree.set_yticks(range(len(lanes)))
    ax_tree.set_yticklabels([b.split("/")[-1] for b in lanes], fontsize=8)
    ax_tree.set_ylabel("idea lanes (one per agent)")
    ax_tree.set_title("Idea tree — each agent explores its own branch; green ring = raised team best",
                      fontsize=10, color=dgreen, weight="bold")
    ax_tree.margins(y=0.3)

    # -- bottom: team-avg hillclimb --
    ax_hc.fill_between(team_t, team_avg, step="post", color=green, alpha=0.30)
    ax_hc.step(team_t, team_avg, where="post", color=dgreen, lw=2)
    ax_hc.set_ylim(0, 109)
    ax_hc.set_yticks([0, 50, 100]); ax_hc.set_yticklabels(["0", "50%", "100%"])
    ax_hc.axhline(100, ls=":", color="grey", lw=0.8)
    ax_hc.set_ylabel("team-avg best success")
    ax_hc.set_xlabel(f"research wall-clock time ({time_unit})")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    n_rings = sum(1 for *_, ring in nodes if ring)
    print(f"wrote {out}  ({len(rows)} ideas across {len(lanes)} lanes, "
          f"{n_rings} improvements, team-avg best={max(team_avg):.1f}%)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="ENPIRE Figure 1 (idea tree + hillclimb).")
    p.add_argument("--out", default="enpire_sim/reports/figure1.png")
    p.add_argument("--time-unit", choices=["min", "h"], default="min")
    p.add_argument("--logs", nargs="*", default=None, help="explicit iteration_log paths")
    args = p.parse_args()
    logs = args.logs if args.logs else default_logs()
    print(f"reading {len(logs)} station log(s)")
    figure1(collect(logs), args.out, time_unit=args.time_unit)


if __name__ == "__main__":
    main()
