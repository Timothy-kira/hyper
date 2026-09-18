"""Discovery trees for the Dream-RSI loop (arXiv:2609.14858, §3).

A tree is rooted at the initial workspace. Every non-root node records one
generation-evaluation attempt: the candidate it produced, the evaluator's
score and diagnostics, and what it cost. The tree is the unit that later
becomes a replay simulator, so nodes persist their full observation.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = "root"


@dataclass
class Node:
    """One generation-evaluation attempt."""

    id: str
    parent: str | None
    round: int                      # decision round that scheduled this attempt
    candidate: dict = field(default_factory=dict)   # the proposed solution config
    proposal: str = ""              # agent's rationale, read by later siblings
    score: float | None = None      # evaluator score; None until evaluated
    diagnostics: dict = field(default_factory=dict)
    cost_seconds: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.error is None and self.score is not None


class DiscoveryTree:
    """Append-only tree of attempts, persisted as one JSON file."""

    def __init__(self, tree_id: str | None = None, meta: dict | None = None):
        self.id = tree_id or uuid.uuid4().hex[:8]
        self.meta = meta or {}
        self.nodes: dict[str, Node] = {}
        self._children: dict[str, list[str]] = {ROOT: []}

    # -- construction -----------------------------------------------------
    def add(self, parent: str, round_: int, **kw) -> Node:
        if parent != ROOT and parent not in self.nodes:
            raise KeyError(f"unknown parent {parent!r}")
        node = Node(id=uuid.uuid4().hex[:8], parent=parent, round=round_, **kw)
        self.nodes[node.id] = node
        self._children.setdefault(parent, []).append(node.id)
        self._children.setdefault(node.id, [])
        return node

    def children(self, node_id: str) -> list[str]:
        return list(self._children.get(node_id, ()))

    # -- queries ----------------------------------------------------------
    def eligible(self) -> list[str]:
        """A(T) = {root} u {leaves} — where exploration may continue."""
        return [ROOT] + [n for n in self.nodes if not self._children.get(n)]

    def best(self) -> Node | None:
        scored = [n for n in self.nodes.values() if n.ok]
        return max(scored, key=lambda n: n.score) if scored else None

    def path_to(self, node_id: str) -> list[Node]:
        """Ancestor chain root->node, the context a resuming agent inherits."""
        chain, cur = [], node_id
        while cur and cur != ROOT:
            chain.append(self.nodes[cur])
            cur = self.nodes[cur].parent
        return list(reversed(chain))

    @property
    def n_attempts(self) -> int:
        return len(self.nodes)

    @property
    def total_cost(self) -> float:
        return sum(n.cost_seconds for n in self.nodes.values())

    # -- persistence ------------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "id": self.id,
            "meta": self.meta,
            "nodes": [asdict(n) for n in self.nodes.values()],
        }, indent=2))
        return path

    @classmethod
    def load(cls, path) -> "DiscoveryTree":
        d = json.loads(Path(path).read_text())
        t = cls(d["id"], d.get("meta"))
        for nd in d["nodes"]:
            n = Node(**nd)
            t.nodes[n.id] = n
            t._children.setdefault(n.parent, []).append(n.id)
            t._children.setdefault(n.id, [])
        return t


def load_pool(directory) -> list[DiscoveryTree]:
    """Load every recorded tree — the simulator pool the dreamer replays."""
    return [DiscoveryTree.load(p) for p in sorted(Path(directory).glob("tree_*.json"))]
