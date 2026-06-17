"""Policy registry.

A *policy* is a callable ``policy(obs) -> np.ndarray`` (a 2-vector target position
in [0, 512]^2), optionally with a ``reset()`` method called at episode start. The
rollout harness only relies on this protocol, so heuristic, CEM, BC, or RL policies
are all interchangeable.

``make_policy(name, **kwargs)`` builds a fresh instance -- the rollout harness uses
this as a *factory* so each parallel worker constructs its own policy (important for
sim-in-the-loop policies like CEM that own a private env that cannot be pickled).
"""
from __future__ import annotations

from typing import Callable

# Factories: name -> (**kwargs) -> policy instance
_REGISTRY: dict[str, Callable[..., object]] = {}


def register(name: str):
    def deco(factory: Callable[..., object]):
        _REGISTRY[name] = factory
        return factory
    return deco


def make_policy(name: str, **kwargs) -> object:
    if name not in _REGISTRY:
        raise KeyError(f"unknown policy '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


@register("cem")
def _make_cem(**kwargs):
    from enpire_sim.policies.baseline_cem import CEMPolicy
    return CEMPolicy(**kwargs)


@register("policy")
def _make_editable(**kwargs):
    from enpire_sim.policies.policy import Policy
    return Policy(**kwargs)
