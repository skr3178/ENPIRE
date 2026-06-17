"""CEM-MPC baseline policy for gym_pusht/PushT-v0.

Stitched from the ENPIRE agent-written code recovered from the project website
(``pusht_components/agent_code/recovered_policies_ordered.py``). This is the
"Claude Code" strict-API policy: it NEVER inspects or mutates the eval env. It
owns a private sim env (via gym.make) that it syncs to each observation through
the public ``reset(options={"reset_to_state": ...})`` API, with the CoG-quirk
compensation, then runs Cross-Entropy-Method MPC over action sequences.

Only change from the recovered source: the two ``gym.make(...)`` calls (which were
truncated at template-interpolation seams in the bundle) are filled with the real
env id, and ``compensated_reset_state`` is imported from ``envs.init_sampler``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from enpire_sim.envs.init_sampler import compensated_reset_state

GOAL_POS = np.array([256.0, 256.0])
GOAL_ANGLE = math.pi / 4
COG_LOCAL = np.array([0.0, 45.0])  # body-frame centre-of-gravity offset

# T outline (used only by the bias-trajectory contact picker; no env internals).
T_POLY_LOCAL = np.array(
    [
        [-60.0, 0.0], [60.0, 0.0], [60.0, 30.0], [15.0, 30.0],
        [15.0, 120.0], [-15.0, 120.0], [-15.0, 30.0], [-60.0, 30.0],
    ]
)


def _build_edges(poly: np.ndarray):
    edges = []
    n = len(poly)
    centroid = poly.mean(axis=0)
    for i in range(n):
        p0, p1 = poly[i], poly[(i + 1) % n]
        edge = p1 - p0
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)
        mid = 0.5 * (p0 + p1)
        if np.dot(normal, mid - centroid) < 0:
            normal = -normal
        edges.append((p0, p1, normal))
    return edges


def _candidate_contacts_local(n_per_edge: int = 3):
    out = []
    for p0, p1, n in _build_edges(T_POLY_LOCAL):
        for k in range(n_per_edge):
            t = (k + 1) / (n_per_edge + 1)
            p = (1 - t) * p0 + t * p1
            out.append((p, -n))  # inward push direction
    return out


CONTACTS_LOCAL = _candidate_contacts_local(n_per_edge=3)


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def clip_action(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 512.0).astype(np.float32)


def shaped_cost(block_pos: np.ndarray, theta: float) -> float:
    pos_err = float(np.linalg.norm(GOAL_POS - block_pos))
    angle_err = abs(wrap_to_pi(GOAL_ANGLE - theta))
    return pos_err + 100.0 * angle_err


def _bias_trajectory(obs: np.ndarray, horizon: int, rng: np.random.Generator) -> np.ndarray:
    """Action-sequence prior used to seed CEM's first sample."""
    agent = np.asarray(obs[:2], dtype=np.float32)
    block = np.asarray(obs[2:4], dtype=np.float32)
    theta = float(obs[4])
    R = rot(theta)

    desired = GOAL_POS - block
    dn = np.linalg.norm(desired)
    desired = desired / dn if dn > 1e-6 else np.array([1.0, 0.0])

    best_idx, best_score = 0, -1e9
    for idx, (p_local, inward_local) in enumerate(CONTACTS_LOCAL):
        inward_world = R @ inward_local
        inward_world = inward_world / max(np.linalg.norm(inward_world), 1e-9)
        score = float(np.dot(inward_world, desired)) + 0.3 * rng.standard_normal()
        if score > best_score:
            best_score, best_idx = score, idx
    p_local, inward_local = CONTACTS_LOCAL[best_idx]
    contact = block + R @ p_local
    inward = R @ inward_local
    inward = inward / max(np.linalg.norm(inward), 1e-9)

    goal_cog = GOAL_POS + np.array(
        [-COG_LOCAL[1] * math.sin(GOAL_ANGLE), COG_LOCAL[1] * math.cos(GOAL_ANGLE)]
    )
    post = contact + inward * 30.0
    anchor = 0.4 * post + 0.6 * goal_cog

    h1 = max(1, horizon // 3)
    seq = np.zeros((horizon, 2), dtype=np.float32)
    for t in range(horizon):
        if t < h1:
            a = (t + 1) / h1
            seq[t] = (1 - a) * agent + a * contact
        else:
            b = (t - h1 + 1) / max(1, horizon - h1)
            seq[t] = (1 - b) * contact + b * anchor
    return np.clip(seq, 0.0, 512.0)


@dataclass
class CEMPolicy:
    """CEM-MPC over action sequences.

    Real (eval) env access: ONLY env.step()/env.reset() in the driver. The policy
    itself receives the observation array and never touches the eval env. It owns a
    private sim env synced to the observation each call (poses via the public reset
    API + compensation; velocities via finite-difference written onto the sim body).
    """

    horizon: int = 25
    n_samples: int = 60
    n_iters: int = 2
    elite_frac: float = 0.2
    init_std: float = 70.0
    min_std: float = 12.0
    max_std: float = 120.0
    warm_mix: float = 0.6
    n_bias_samples: int = 4
    pos_tol: float = 1.0
    angle_tol: float = 0.02
    retreat_dist: float = 70.0
    seed: int = 0
    env_dt: float = 0.1  # seconds per env.step (control_hz=10)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.sim_env = None
        self.prev_mean: Optional[np.ndarray] = None
        self.prev_obs: Optional[np.ndarray] = None

    def reset(self):
        self.prev_mean = None
        self.prev_obs = None
        self.rng = np.random.default_rng(self.seed)

    def _ensure_sim(self):
        if self.sim_env is None:
            import gymnasium as gym
            import gym_pusht  # noqa: F401
            self.sim_env = gym.make("gym_pusht/PushT-v0", obs_type="state")
            self.sim_env.reset(seed=12345)

    def _estimate_velocities(self, obs: np.ndarray):
        if self.prev_obs is None:
            return None
        dt = self.env_dt
        prev = self.prev_obs
        agent_vel = (obs[:2] - prev[:2]) / dt
        block_vel = (obs[2:4] - prev[2:4]) / dt
        dtheta = wrap_to_pi(obs[4] - prev[4])
        block_angvel = dtheta / dt
        return agent_vel, block_vel, block_angvel

    def _sync_sim(self, obs: np.ndarray):
        sim = self.sim_env
        sim.reset(options={"reset_to_state": compensated_reset_state(obs)})
        vel = self._estimate_velocities(obs)
        if vel is not None:
            agent_vel, block_vel, block_angvel = vel
            uw = sim.unwrapped  # owned by us -- not the eval env
            uw.agent.velocity = (float(agent_vel[0]), float(agent_vel[1]))
            uw.block.velocity = (float(block_vel[0]), float(block_vel[1]))
            uw.block.angular_velocity = float(block_angvel)

    def _score(self, obs, seq: np.ndarray) -> float:
        sim = self.sim_env
        self._sync_sim(obs)
        min_cost = shaped_cost(np.asarray(obs[2:4]), float(obs[4]))
        success = False
        for t in range(seq.shape[0]):
            o, _, terminated, _, _ = sim.step(seq[t])
            c = shaped_cost(np.asarray(o[2:4]), float(o[4]))
            if c < min_cost:
                min_cost = c
            if terminated:
                success = True
                break
        return -min_cost + (1000.0 if success else 0.0)

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        agent = obs[:2]
        block = obs[2:4]
        theta = float(obs[4])
        angle_err = wrap_to_pi(GOAL_ANGLE - theta)
        pos_err_mag = float(np.linalg.norm(GOAL_POS - block))

        if abs(angle_err) <= self.angle_tol and pos_err_mag <= self.pos_tol:
            cog = block + rot(theta) @ COG_LOCAL
            away = agent - cog
            if np.linalg.norm(away) < 1e-6:
                away = np.array([0.0, 1.0])
            away = away / np.linalg.norm(away)
            self.prev_obs = obs.copy()
            return clip_action(cog + away * self.retreat_dist)

        self._ensure_sim()

        H, N = self.horizon, self.n_samples
        n_elite = max(2, int(self.elite_frac * N))

        if self.prev_mean is not None:
            shifted = np.vstack([self.prev_mean[1:], self.prev_mean[-1:]])
            bias = _bias_trajectory(obs, H, self.rng)
            mean = self.warm_mix * shifted + (1 - self.warm_mix) * bias
        else:
            mean = _bias_trajectory(obs, H, self.rng)
        std = np.full((H, 2), self.init_std, dtype=np.float32)

        best_seq = mean.copy()
        best_score = -1e18
        for it in range(self.n_iters):
            samples = np.empty((N, H, 2), dtype=np.float32)
            if it == 0:
                pool = [mean]
                for _ in range(max(1, self.n_bias_samples - 1)):
                    pool.append(_bias_trajectory(obs, H, self.rng))
                for i in range(N):
                    base = pool[i % len(pool)]
                    samples[i] = base + self.rng.normal(0.0, self.init_std, size=(H, 2))
            else:
                samples = mean + self.rng.normal(0.0, std, size=(N, H, 2))
            samples = np.clip(samples, 0.0, 512.0)

            scores = np.empty(N)
            for i in range(N):
                scores[i] = self._score(obs, samples[i])

            order = np.argsort(scores)
            elite_idx = order[-n_elite:]
            elites = samples[elite_idx]
            top_score = float(scores[order[-1]])
            if top_score > best_score:
                best_score = top_score
                best_seq = samples[order[-1]].copy()

            mean = elites.mean(axis=0)
            std = np.clip(elites.std(axis=0), self.min_std, self.max_std)

        self.prev_mean = best_seq
        self.prev_obs = obs.copy()
        return clip_action(best_seq[0])
