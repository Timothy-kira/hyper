"""Exploration policies — the only component Dream-RSI rewrites.

The discovery agent, the evaluator and the execution interfaces stay fixed;
self-improvement happens entirely in the code below, which decides where to
continue exploring, how many attempts to run in parallel, and when to stop.

A policy sees a ``TreeView`` and returns a batch of node ids to extend. A node
may appear more than once: each occurrence starts one attempt from it, which
is how several branches are opened off the root in a single round.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict

from .tree import ROOT


class ExplorationPolicy:
    """Interface shared by online rollouts and replay."""

    def reset(self) -> None:  # per-rollout state
        pass

    def select(self, view, workers: int) -> list[str]:
        raise NotImplementedError


@dataclass
class PolicyParams:
    """The meta-search space the dreamer optimizes over."""

    n_branches: int = 2          # independent workspaces kept alive
    max_depth: int = 6           # refinements before a branch is abandoned
    greedy_workers: int = 0      # workers always aimed at the global best leaf
    softmax_temp: float = 0.05   # 0 = argmax; higher = flatter leaf sampling
    stall_patience: int = 3      # rounds without gain before reopening a branch
    stop_patience: int = 6       # rounds without gain before stopping entirely
    seed: int = 0

    def clamp(self) -> "PolicyParams":
        self.n_branches = max(1, min(8, int(self.n_branches)))
        self.max_depth = max(1, min(20, int(self.max_depth)))
        self.greedy_workers = max(0, min(4, int(self.greedy_workers)))
        self.softmax_temp = max(0.0, min(1.0, float(self.softmax_temp)))
        self.stall_patience = max(1, min(10, int(self.stall_patience)))
        self.stop_patience = max(1, min(24, int(self.stop_patience)))
        return self

    def to_dict(self) -> dict:
        return asdict(self)


class ParallelRefine(ExplorationPolicy):
    """Parameterized parallel-refining search.

    The paper's manually designed starting policy keeps several independent
    workspaces and repeatedly refines each one against its own local history.
    This generalizes that: branches still refine independently, but the policy
    can also reserve workers to exploit the globally best leaf, abandon a
    stalled branch back to the root, and stop early once gains dry up.
    ``PolicyParams()`` defaults reproduce plain parallel refining.
    """

    def __init__(self, params: PolicyParams | dict | None = None):
        if isinstance(params, dict):
            params = PolicyParams(**params)
        self.p = (params or PolicyParams()).clamp()
        self.reset()

    def reset(self) -> None:
        self._rng = random.Random(self.p.seed)
        self._tips: list[str] = []      # current leaf of each live branch
        self._best = float("-inf")
        self._stall = 0                 # rounds since the last global gain
        self._branch_stall: dict[str, int] = {}

    def _pick_leaf(self, view, candidates: list[str]) -> str:
        """Argmax over scores, or a softmax sample when the temperature is on."""
        scored = [(c, view.score(c)) for c in candidates]
        scored = [(c, s if s is not None else 0.0) for c, s in scored]
        if self.p.softmax_temp <= 0:
            return max(scored, key=lambda t: t[1])[0]
        hi = max(s for _, s in scored)
        w = [math.exp((s - hi) / self.p.softmax_temp) for _, s in scored]
        return self._rng.choices([c for c, _ in scored], weights=w, k=1)[0]

    def select(self, view, workers: int) -> list[str]:
        eligible = set(view.eligible())
        if not eligible:
            return []

        best = view.best_so_far()
        if best > self._best:
            self._best, self._stall = best, 0
        else:
            self._stall += 1
        if self._stall >= self.p.stop_patience:
            return []  # nothing is improving; end the rollout

        # Drop branch tips that died, went too deep, or stalled locally.
        live = []
        for t in self._tips:
            if t not in eligible or view.depth(t) >= self.p.max_depth:
                continue
            if self._branch_stall.get(t, 0) >= self.p.stall_patience:
                continue
            live.append(t)
        self._tips = live

        batch: list[str] = []

        # Exploit: aim reserved workers at the best revealed leaf.
        leaves = [n for n in eligible if n != ROOT]
        for _ in range(min(self.p.greedy_workers, workers)):
            if not leaves:
                break
            batch.append(self._pick_leaf(view, leaves))

        # Refine each live branch, then open new ones off the root up to n_branches.
        for tip in self._tips:
            if len(batch) >= workers:
                break
            batch.append(tip)
        while (len(batch) < workers
               and len(self._tips) < self.p.n_branches
               and ROOT in eligible):
            batch.append(ROOT)
            self._tips.append(ROOT)  # replaced by the real child id after the round

        # Spend any idle worker on the best leaf available.
        while len(batch) < workers and leaves:
            batch.append(self._pick_leaf(view, leaves))

        return batch[:workers]

    def observe(self, produced: list[tuple[str, str]], view) -> None:
        """Advance branch tips onto the children the round actually produced.

        ``produced`` is ordered and may name the same parent twice, so tips are
        matched positionally against the attempts this policy scheduled.
        """
        pending: dict[str, list[str]] = {}
        for parent, child in produced:
            pending.setdefault(parent, []).append(child)

        tips = []
        for tip in self._tips:
            queue = pending.get(tip)
            if not queue:
                tips.append(tip)
                continue
            child = queue.pop(0)
            ps, cs = view.score(tip), view.score(child)
            if ps is not None and cs is not None and cs <= ps:
                self._branch_stall[child] = self._branch_stall.get(tip, 0) + 1
            tips.append(child)
        # Keep only distinct live tips; ROOT placeholders are now real nodes.
        self._tips = [t for t in dict.fromkeys(tips) if t != ROOT]


def build(spec: dict | None) -> ExplorationPolicy:
    """Instantiate a policy from a serialized spec."""
    spec = spec or {}
    kind = spec.get("kind", "parallel_refine")
    if kind != "parallel_refine":
        raise ValueError(f"unknown policy kind {kind!r}")
    return ParallelRefine(spec.get("params", {}))
