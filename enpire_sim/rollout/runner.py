"""R module: budgeted rollouts that preserve state/action/video/result for audit.

`run_episode` runs one policy on one seeded initial state through the (immutable)
PushTInterface, optionally recording a video + trajectory. `evaluate` runs a set of
seeds -- optionally across worker processes -- and returns an audit-friendly summary.

Policies are passed as a *spec* ``(name, kwargs)`` resolved by the registry inside
each worker, so sim-in-the-loop policies (which own an unpicklable env) construct
fresh per worker. Supports both obs-only policies ``policy(obs)`` and env-needing
ones ``policy(obs, env=...)`` (e.g. the Codex beam-search policy).
"""
from __future__ import annotations

import inspect
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from enpire_sim.envs.pusht_interface import PushTInterface, EnvConfig
from enpire_sim.policies import make_policy
from enpire_sim.rollout import recorder


@dataclass
class RolloutConfig:
    max_steps: int = 300
    record_video: bool = False
    record_trajectory: bool = False
    out_dir: str = "enpire_sim/reports/rollouts"
    video_prefix: str = "policy"
    fps: int = 10


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    coverage: float
    steps: int
    reward_final: float
    wall_time: float
    video_path: Optional[str] = None
    trajectory_path: Optional[str] = None


def _policy_wants_env(policy) -> bool:
    try:
        return "env" in inspect.signature(policy.__call__).parameters
    except (TypeError, ValueError):
        return False


def run_episode(policy, seed: int, config: RolloutConfig,
                interface: Optional[PushTInterface] = None) -> EpisodeResult:
    """Run one episode. Reuses `interface` if given, else builds one."""
    own_iface = interface is None
    if own_iface:
        interface = PushTInterface(EnvConfig(max_steps=config.max_steps,
                                             render=config.record_video))
    wants_env = _policy_wants_env(policy)
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "set_seed"):
        policy.set_seed(seed)

    obs, info = interface.auto_reset(seed=seed)
    frames: List[np.ndarray] = []
    observations, actions, coverages = [obs.copy()], [], [float(info.get("coverage", 0.0))]
    if config.record_video:
        frames.append(interface.frame())

    t0 = time.perf_counter()
    terminated = truncated = False
    while not (terminated or truncated) and interface.t < config.max_steps:
        action = policy(obs, env=interface.env) if wants_env else policy(obs)
        obs, reward, terminated, truncated, info = interface.step(action)
        observations.append(obs.copy())
        actions.append(np.asarray(action, dtype=np.float32))
        coverages.append(float(info.get("coverage", 0.0)))
        if config.record_video:
            frames.append(interface.frame())
    wall = time.perf_counter() - t0

    success = interface.success
    steps = interface.t
    final_cov = interface.coverage

    video_path = trajectory_path = None
    if config.record_video:
        path = f"{config.out_dir}/{recorder.video_name(config.video_prefix, seed, success)}"
        video_path = recorder.save_video(frames, path, fps=config.fps)
    if config.record_trajectory:
        tpath = f"{config.out_dir}/{config.video_prefix}_seed{seed:03d}.npz"
        trajectory_path = recorder.save_trajectory(
            tpath, np.asarray(observations), np.asarray(actions), np.asarray(coverages))

    if own_iface:
        interface.close()

    return EpisodeResult(
        seed=seed, success=success, coverage=final_cov, steps=steps,
        reward_final=float(reward) if actions else 0.0, wall_time=wall,
        video_path=video_path, trajectory_path=trajectory_path,
    )


def _worker(spec: Tuple[str, dict], seed: int, config: RolloutConfig) -> EpisodeResult:
    name, kwargs = spec
    policy = make_policy(name, **kwargs)
    return run_episode(policy, seed, config)


def evaluate(policy_spec: Tuple[str, dict], seeds: List[int],
             config: Optional[RolloutConfig] = None, workers: int = 1) -> Dict[str, Any]:
    """Evaluate a policy over `seeds`. Returns a summary + per-episode results."""
    config = config or RolloutConfig()
    results: List[EpisodeResult] = []

    if workers <= 1:
        name, kwargs = policy_spec
        for s in seeds:
            results.append(run_episode(make_policy(name, **kwargs), s, config))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, policy_spec, s, config): s for s in seeds}
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: r.seed)
    n = len(results)
    n_success = sum(r.success for r in results)
    return {
        "policy": policy_spec[0],
        "n_episodes": n,
        "n_success": n_success,
        "success_rate": n_success / n if n else 0.0,
        "mean_coverage": float(np.mean([r.coverage for r in results])) if n else 0.0,
        "mean_steps": float(np.mean([r.steps for r in results])) if n else 0.0,
        "total_wall_time": float(sum(r.wall_time for r in results)),
        "episodes": [asdict(r) for r in results],
    }
