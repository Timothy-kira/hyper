"""Replay simulator: a completed discovery tree reused as a world model.

Dream-RSI's premise (arXiv:2609.14858, §2) is that a finished discovery run
already records every outcome along every branch it explored. Re-walking that
record lets an alternative exploration policy be scored without rerunning the
coding agent or the evaluator, turning one expensive online rollout into
thousands of zero-execution-cost off-policy evaluations.

Online rollouts and replays drive the policy through the same TreeView
interface, so a policy improved by dreaming can be redeployed unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tree import ROOT, DiscoveryTree, Node


class TreeView:
    """What an exploration policy is allowed to see when it decides.

    Identical in shape online and in replay: online the revealed set is the
    whole tree built so far, in replay it is the subtree uncovered by this
    policy's own choices.
    """

    def __init__(self, tree: DiscoveryTree, revealed: set[str], round_: int,
                 online: bool = False):
        self._t, self._revealed, self.round = tree, revealed, round_
        # Online, the discovery agent can always produce another child, so every
        # revealed leaf stays extendable. In replay only recorded children exist.
        self._online = online

    def eligible(self) -> list[str]:
        """Root, plus revealed nodes that can still yield a new attempt."""
        out = [ROOT] if self._can_extend(ROOT) else []
        for nid in self._revealed:
            if self._is_revealed_leaf(nid) and self._can_extend(nid):
                out.append(nid)
        return out

    def _is_revealed_leaf(self, nid: str) -> bool:
        return not any(c in self._revealed for c in self._t.children(nid))

    def _can_extend(self, nid: str) -> bool:
        if self._online:
            return True
        return any(c not in self._revealed for c in self._t.children(nid))

    def node(self, nid: str) -> Node | None:
        return self._t.nodes.get(nid)

    def score(self, nid: str) -> float | None:
        n = self._t.nodes.get(nid)
        return n.score if n and n.ok else None

    def depth(self, nid: str) -> int:
        return 0 if nid == ROOT else len(self._t.path_to(nid))

    def revealed(self) -> list[str]:
        return sorted(self._revealed)

    def best_so_far(self) -> float:
        s = [self.score(n) for n in self._revealed]
        s = [x for x in s if x is not None]
        return max(s) if s else float("-inf")

    def siblings(self, nid: str) -> list[str]:
        n = self._t.nodes.get(nid)
        if not n:
            return []
        return [c for c in self._t.children(n.parent) if c != nid and c in self._revealed]


@dataclass
class ReplayResult:
    best_score: float
    n_generations: int
    n_rounds: int
    objective: float

    @property
    def attempts_per_round(self) -> float:
        return self.n_generations / self.n_rounds if self.n_rounds else 0.0


class ReplaySimulator:
    """Replays one recorded tree under an alternative exploration policy."""

    def __init__(self, tree: DiscoveryTree):
        self.tree = tree

    def run(self, policy, workers: int = 2, max_rounds: int = 32,
            cost_weight: float = 0.05, parallel_weight: float = 0.02) -> ReplayResult:
        """Drive ``policy`` through the recorded tree and score its trajectory.

        Selecting a node reveals one of its recorded-but-unrevealed children;
        a node with none left simply stops being eligible, so a policy is never
        charged for a branch it could not know was exhausted.
        """
        if hasattr(policy, "reset"):
            policy.reset()

        revealed: set[str] = set()
        n_gen = n_rounds = 0

        for k in range(max_rounds):
            view = TreeView(self.tree, revealed, k)
            if not view.eligible():
                break
            batch = policy.select(view, workers) or []
            # Honour the interface contract rather than trusting the policy.
            # A node may repeat: each occurrence is one attempt started from it,
            # which is how the base policy opens several branches off the root
            # in a single round.
            allowed = set(view.eligible())
            batch = [n for n in batch if n in allowed][:workers]
            if not batch:
                break
            n_rounds += 1
            produced: list[tuple[str, str]] = []
            for nid in batch:
                pending = [c for c in self.tree.children(nid) if c not in revealed]
                if not pending:
                    continue
                revealed.add(pending[0])
                produced.append((nid, pending[0]))
                n_gen += 1
            if hasattr(policy, "observe"):
                policy.observe(produced, TreeView(self.tree, revealed, k))

        best = max((self.tree.nodes[n].score for n in revealed
                    if self.tree.nodes[n].ok), default=0.0)
        total = max(1, self.tree.n_attempts)
        # Paper's replay objective: solution quality, less the cost of the
        # generations spent, plus a bonus for batching them into fewer rounds.
        obj = (best
               - cost_weight * (n_gen / total)
               + parallel_weight * ((n_gen / n_rounds) / workers if n_rounds else 0.0))
        return ReplayResult(best, n_gen, n_rounds, obj)


def score_policy(policy_factory, pool: list[DiscoveryTree], **kw) -> dict:
    """Average a policy's replay objective across the whole simulator pool."""
    results = [ReplaySimulator(t).run(policy_factory(), **kw) for t in pool]
    n = max(1, len(results))
    return {
        "objective": sum(r.objective for r in results) / n,
        "best_score": sum(r.best_score for r in results) / n,
        "generations": sum(r.n_generations for r in results) / n,
        "rounds": sum(r.n_rounds for r in results) / n,
        "per_tree": [r.__dict__ for r in results],
    }
