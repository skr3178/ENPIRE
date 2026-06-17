"""Stage 0/1 entry point.

Eval mode (Stage 0): evaluate a fixed policy over N seeded episodes and write a
results.json + (optionally) videos. This is the harness smoke test.

    python -m enpire_sim.run_single --policy cem --episodes 50 --eval-only
    python -m enpire_sim.run_single --policy policy --episodes 20 --videos

The agent-loop mode (Stage 1) is added later in agents/coding_agent.py.
"""
from __future__ import annotations

import argparse
import json
import os

# CEM/rollouts are CPU-bound and embarrassingly parallel across episodes. Cap at 24
# workers (of 32 cores) to leave headroom for the rest of the machine. Override with
# --workers.
DEFAULT_WORKERS = min(24, os.cpu_count() or 1)

from enpire_sim.envs.init_sampler import episode_seeds
from enpire_sim.policies import available
from enpire_sim.rollout.runner import RolloutConfig, evaluate


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a Push-T policy through the ENPIRE harness.")
    p.add_argument("--policy", default="cem", help=f"policy name; one of {available()}")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"parallel episode workers (default: all {DEFAULT_WORKERS} cores)")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--videos", action="store_true", help="record a video per episode")
    p.add_argument("--trajectories", action="store_true", help="save per-episode .npz")
    p.add_argument("--eval-only", action="store_true", help="(default behavior; kept for clarity)")
    p.add_argument("--out", default="enpire_sim/reports")
    args = p.parse_args()

    seeds = episode_seeds(args.episodes, base=args.base_seed)
    config = RolloutConfig(
        max_steps=args.max_steps,
        record_video=args.videos,
        record_trajectory=args.trajectories,
        out_dir=os.path.join(args.out, "rollouts"),
        video_prefix=args.policy,
    )

    print(f"Evaluating policy '{args.policy}' over {len(seeds)} episodes "
          f"(seeds {seeds[0]}..{seeds[-1]}, workers={args.workers})...")
    summary = evaluate((args.policy, {}), seeds, config=config, workers=args.workers)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"results_{args.policy}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  success rate : {summary['success_rate']*100:.1f}%  "
          f"({summary['n_success']}/{summary['n_episodes']})")
    print(f"  mean coverage: {summary['mean_coverage']:.3f}")
    print(f"  mean steps   : {summary['mean_steps']:.1f}")
    print(f"  wall time    : {summary['total_wall_time']:.1f}s")
    print(f"  wrote        : {out_path}")


if __name__ == "__main__":
    main()
