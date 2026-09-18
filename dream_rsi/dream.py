"""Dreaming: offline policy improvement over the replay pool.

Dream-RSI's offline phase (arXiv:2609.14858, §3) holds the history fixed and
develops M policy versions in sequence, starting from the currently deployed
one. Each version is replayed across every recorded tree before the next is
proposed, so proposals are informed by measured feedback rather than guesswork.
Because replay costs nothing to execute, many versions can be tried before a
single GPU-hour is spent online.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .policy import ParallelRefine, PolicyParams
from .replay import score_policy
from .tree import DiscoveryTree


@dataclass
class Version:
    spec: dict
    metrics: dict
    parent: int | None
    index: int

    @property
    def objective(self) -> float:
        return self.metrics["objective"]


# Coordinate moves over the meta-search space. A proposer that only mutates
# parameters is enough while the policy family is parameterized; the same
# interface accepts an LLM that rewrites policy.py outright.
_MOVES = {
    "n_branches": [-2, -1, 1, 2],
    "max_depth": [-4, -2, 2, 4],
    "greedy_workers": [-1, 1],
    "stall_patience": [-2, -1, 1, 2],
    "stop_patience": [-3, 3, 6],
    "softmax_temp": [-0.04, -0.01, 0.01, 0.04],
}


class MutationProposer:
    """Proposes the next policy version from the versions already measured.

    Hill-climbs from the best version so far, biased away from moves that have
    already been tried and toward coordinates not yet perturbed — the dreamer's
    stand-in for an LLM reading the replay feedback.
    """

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._tried: set[tuple] = set()

    def propose(self, history: list[Version]) -> dict | None:
        best = max(history, key=lambda v: v.objective)
        base = PolicyParams(**best.spec["params"])
        for _ in range(64):
            key = self._rng.choice(list(_MOVES))
            delta = self._rng.choice(_MOVES[key])
            cand = PolicyParams(**base.to_dict())
            setattr(cand, key, getattr(cand, key) + delta)
            cand.clamp()
            sig = tuple(sorted(cand.to_dict().items()))
            if sig in self._tried or sig == tuple(sorted(base.to_dict().items())):
                continue
            self._tried.add(sig)
            return {"kind": "parallel_refine", "params": cand.to_dict()}
        return None


def dream(pool: list[DiscoveryTree], start_spec: dict, n_versions: int = 40,
          proposer=None, workers: int = 2, log_path=None, **replay_kw) -> dict:
    """Develop and replay-evaluate policy versions; return the best one.

    Returns the winning spec plus the full version history, so a redeployment
    can be traced back to the replay evidence that chose it.
    """
    if not pool:
        raise ValueError("cannot dream with an empty simulator pool")
    proposer = proposer or MutationProposer()

    def factory(spec):
        return lambda: ParallelRefine(spec["params"])

    history: list[Version] = []
    metrics = score_policy(factory(start_spec), pool, workers=workers, **replay_kw)
    history.append(Version(start_spec, metrics, None, 0))

    for i in range(1, n_versions):
        spec = proposer.propose(history)
        if spec is None:
            break
        m = score_policy(factory(spec), pool, workers=workers, **replay_kw)
        parent = max(range(len(history)), key=lambda j: history[j].objective)
        history.append(Version(spec, m, parent, i))

    best = max(history, key=lambda v: v.objective)
    incumbent = history[0]
    out = {
        "best_spec": best.spec,
        "best_metrics": best.metrics,
        "incumbent_metrics": incumbent.metrics,
        "improved": best.index != 0,
        "gain": best.objective - incumbent.objective,
        "n_versions": len(history),
        "pool_size": len(pool),
        "history": [
            {"index": v.index, "parent": v.parent, "params": v.spec["params"],
             "objective": v.objective, "best_score": v.metrics["best_score"],
             "generations": v.metrics["generations"], "rounds": v.metrics["rounds"]}
            for v in history
        ],
    }
    if log_path:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
    return out
