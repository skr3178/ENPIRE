"""Reproduction plots (Stage 3).

`hillclimb` reproduces the ENPIRE "HILLCLIMB TIMELINE": best-success-rate-so-far vs
research wall-clock time, as a monotonic step function, with each new best annotated
by what the agent changed and the point gain (+pp). Data source: the autoresearch
iteration log written by enpire_sim.agents.coding_agent.

    python -m enpire_sim.metrics.plots hillclimb --branch autoresearch/run1
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

LOG_JSONL = "enpire_sim/reports/iteration_log.jsonl"


def _load(path: str, branch: Optional[str]) -> List[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if branch is None or r.get("branch") == branch:
                rows.append(r)
    rows.sort(key=lambda r: r.get("ts", 0.0))
    return rows


def hillclimb(rows: List[dict], out: str, title: str = "Hillclimb timeline") -> Optional[str]:
    if not rows:
        print("no iterations logged yet; nothing to plot")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Cumulative research wall-clock (hours) from per-iteration eval time.
    t = []
    acc = 0.0
    for r in rows:
        acc += float(r.get("eval_seconds", 0.0))
        t.append(acc / 3600.0)
    sr = [float(r.get("success_rate", 0.0)) * 100.0 for r in rows]

    # Running best (monotonic) + which iterations set a new best.
    best, best_curve, improvements = -1.0, [], []
    for i, s in enumerate(sr):
        if s > best + 1e-9:
            delta = s - max(best, 0.0)
            improvements.append((i, s, delta))
            best = s
        best_curve.append(best)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    green, dgreen = "#7e9e6b", "#4f6b3f"
    ax.fill_between(t, best_curve, step="post", color=green, alpha=0.30)
    ax.step(t, best_curve, where="post", color=dgreen, lw=2)
    ax.scatter([t[i] for i, _, _ in improvements],
               [s for _, s, _ in improvements], color=dgreen, zorder=5, s=28)

    for i, s, delta in improvements:
        desc = (rows[i].get("description") or "").strip() or "improve"
        ax.annotate(f"{desc}\n+{delta:.1f} pp", xy=(t[i], s),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8, color=dgreen)

    ax.set_xlabel("research wall-clock time (h)")
    ax.set_ylabel("best success rate")
    ax.set_ylim(0, 109)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50%", "100%"])
    ax.axhline(100, ls=":", color="grey", lw=0.8)
    ax.set_title(title, fontsize=11, color=dgreen, weight="bold")
    ax.margins(x=0.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}  ({len(rows)} iterations, best={best:.1f}%)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="ENPIRE reproduction plots.")
    sub = p.add_subparsers(dest="cmd", required=True)
    hc = sub.add_parser("hillclimb")
    hc.add_argument("--log", default=LOG_JSONL)
    hc.add_argument("--branch", default=None, help="filter to one branch (e.g. autoresearch/run1)")
    hc.add_argument("--out", default="enpire_sim/reports/hillclimb.png")
    hc.add_argument("--title", default="Hillclimb timeline")
    args = p.parse_args()
    if args.cmd == "hillclimb":
        hillclimb(_load(args.log, args.branch), args.out, args.title)


if __name__ == "__main__":
    main()
