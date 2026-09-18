"""The outer Dream-RSI loop for HOD26.

Each iteration: deploy the current exploration policy online to grow a fresh
discovery tree, append that tree to the simulator pool, dream over the whole
pool to develop a better policy, then redeploy it. Only the policy changes --
the discovery agent, the evaluator and the candidate space stay fixed, so a
gain between iterations is attributable to exploration, not to a moving target.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from .budget import estimate_round_hours, make_budget
from .candidate import DiscoveryAgent
from .dream import dream
from .executor import KaggleRoundExecutor, LocalMockExecutor
from .policy import build as build_policy
from .replay import TreeView
from .tree import ROOT, DiscoveryTree, load_pool

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "runs"


def online_rollout(policy, agent, executor, *, workers: int, max_rounds: int,
                   round_base: dict, tree_meta: dict, budget: Budget | None = None,
                   save_to: Path | None = None, log=print) -> DiscoveryTree:
    """One online rollout: the policy drives real GPU attempts into a new tree.

    The tree is persisted after every round, not at the end. A rollout costs
    hours of GPU time, and an interruption -- a crash, a restart to pick up a
    changed search space -- should cost the round in flight, not all of them.
    """
    tree = DiscoveryTree(meta=tree_meta)
    costs: list[float] = []
    path = (Path(save_to) / f"tree_{int(time.time())}_{tree.id}.json") if save_to else None

    for k in range(max_rounds):
        if budget is not None:
            need = estimate_round_hours(costs, workers)
            if not budget.affords(need):
                log(f"round {k}: stopping -- next round needs ~{need:.2f} GPU-h, "
                    f"{budget.searchable_hours:.2f} h left outside the "
                    f"{budget.reserve_hours:.1f} h submission reserve")
                break
        view = TreeView(tree, set(tree.nodes), k, online=True)
        batch = policy.select(view, workers) or []
        batch = [n for n in batch if n in set(view.eligible())][:workers]
        if not batch:
            log(f"round {k}: policy stopped")
            break

        history = [{"candidate": n.candidate, "score": n.score}
                   for n in tree.nodes.values()]
        entries, scheduled = [], []
        for parent in batch:
            parent_cand = None if parent == ROOT else tree.nodes[parent].candidate
            cand, why = agent.propose(parent_cand, history)
            node = tree.add(parent, k, candidate=cand, proposal=why)
            entries.append({"node_id": node.id, "candidate": cand})
            scheduled.append((parent, node.id))
            history.append({"candidate": cand, "score": None})

        log(f"round {k}: {len(entries)} attempts -> {[e['node_id'] for e in entries]}")
        results = executor.run_round({**round_base, "round": k, "candidates": entries})

        by_id = {r["node_id"]: r for r in results}
        for _, node_id in scheduled:
            r = by_id.get(node_id)
            node = tree.nodes[node_id]
            if r is None:
                node.error = "no result returned for this attempt"
                continue
            node.score = r.get("score")
            node.diagnostics = r.get("diagnostics", {})
            node.cost_seconds = r.get("cost_seconds", 0.0)
            node.error = r.get("error")
            if node.ok:
                log(f"    {node_id}: mAP={node.score:.4f} "
                    f"({node.candidate['channels']['mode']}, {node.cost_seconds/60:.0f}min)")
            else:
                log(f"    {node_id}: FAILED")

        costs.extend(tree.nodes[n].cost_seconds for _, n in scheduled
                     if tree.nodes[n].cost_seconds)
        if budget is not None:
            spent = sum(tree.nodes[n].cost_seconds for _, n in scheduled) / 3600
            budget = budget.spend(spent)
            log(f"    round cost {spent:.2f} GPU-h | {budget.searchable_hours:.2f} h "
                f"searchable remaining")

        produced = [(p, n) for p, n in scheduled if tree.nodes[n].ok]
        if hasattr(policy, "observe"):
            policy.observe(produced, TreeView(tree, set(tree.nodes), k, online=True))

        if path is not None:
            tree.save(path)

        best = tree.best()
        log(f"    best so far: {best.score:.4f}" if best else "    nothing scored yet")

    if path is not None and not tree.nodes:
        path.unlink(missing_ok=True)      # an empty rollout is not a replay world
    return tree


def iterate(*, iterations: int, workers: int, max_rounds: int, executor,
            state_dir: Path, round_base: dict, n_versions: int,
            reserve_hours: float = 0.0, deadline=None, log=print) -> dict:
    """Run the recursive self-improvement loop and return its final state."""
    state_dir.mkdir(parents=True, exist_ok=True)
    spec_path = state_dir / "policy.json"
    spec = (json.loads(spec_path.read_text()) if spec_path.exists()
            else {"kind": "parallel_refine", "params": {}})

    budget = None
    if reserve_hours > 0:
        budget = make_budget(reserve_hours, needed_by=deadline)
        if budget is None:
            log("could not read the GPU quota; running without a budget guard")
        elif budget.expires_soon:
            log(f"GPU budget: {budget.remaining_hours:.2f} h remaining, and the "
                f"allowance refreshes before the submission is due -- spending all "
                f"of it on the search, since unused hours are lost at the refresh")
        else:
            log(f"GPU budget: {budget.remaining_hours:.2f} h remaining, "
                f"{reserve_hours:.1f} h reserved for the submission fit, "
                f"{budget.searchable_hours:.2f} h searchable")

    summary = []
    for t in range(1, iterations + 1):
        log(f"\n{'='*66}\nITERATION {t}: online explore  (policy={spec['params']})\n{'='*66}")
        tree = online_rollout(
            build_policy(spec), DiscoveryAgent(seed=t), executor,
            workers=workers, max_rounds=max_rounds, round_base=round_base,
            tree_meta={"iteration": t, "policy": spec}, budget=budget,
            save_to=state_dir, log=log)

        best = tree.best()
        log(f"\nrollout: {tree.n_attempts} attempts, "
            f"{tree.total_cost/3600:.2f} GPU-h, best={best.score if best else float('nan'):.4f}")

        pool = load_pool(state_dir)
        log(f"\nITERATION {t}: dreaming over {len(pool)} recorded tree(s), "
            f"{sum(x.n_attempts for x in pool)} nodes")
        d = dream(pool, spec, n_versions=n_versions, workers=workers,
                  log_path=state_dir / f"dream_{t}.json")
        log(f"  dreamed {d['n_versions']} versions | improved={d['improved']} "
            f"| J {d['incumbent_metrics']['objective']:.4f} -> "
            f"{d['best_metrics']['objective']:.4f} ({d['gain']:+.4f})")
        if d["improved"]:
            log(f"  redeploying: {d['best_spec']['params']}")
            spec = d["best_spec"]
            spec_path.write_text(json.dumps(spec, indent=2))

        summary.append({
            "iteration": t, "attempts": tree.n_attempts,
            "gpu_hours": tree.total_cost / 3600,
            "best_score": best.score if best else None,
            "best_candidate": best.candidate if best else None,
            "policy_after": spec["params"], "dream_gain": d["gain"],
        })
        (state_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return {"summary": summary, "policy": spec}


def main() -> None:
    ap = argparse.ArgumentParser(description="Dream-RSI loop for HOD26")
    ap.add_argument("--executor", choices=["kaggle", "mock"], default="mock")
    ap.add_argument("--slug", default="xishengfeng/hod26-round")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3,
                    help="attempts per decision round (one kernel session)")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--versions", type=int, default=40, help="policies per dream")
    ap.add_argument("--proxy-train", type=int, default=600)
    ap.add_argument("--proxy-val", type=int, default=200)
    ap.add_argument("--reserve-hours", type=float, default=7.0,
                    help="GPU hours withheld from the search for the final fit")
    ap.add_argument("--deadline", default="2026-09-24T16:00:00+00:00",
                    help="competition deadline; a quota refresh before it means "
                         "the current allowance is use-it-or-lose-it")
    ap.add_argument("--state-dir", type=Path, default=STATE / "rsi")
    args = ap.parse_args()

    executor = (KaggleRoundExecutor(args.slug) if args.executor == "kaggle"
                else LocalMockExecutor())
    out = iterate(
        iterations=args.iterations, workers=args.workers, max_rounds=args.max_rounds,
        executor=executor, state_dir=args.state_dir, n_versions=args.versions,
        reserve_hours=args.reserve_hours,
        deadline=dt.datetime.fromisoformat(args.deadline),
        round_base={"proxy_train_images": args.proxy_train,
                    "proxy_val_images": args.proxy_val},
    )
    print("\n" + "=" * 66)
    for s in out["summary"]:
        print(f"iter {s['iteration']}: best={s['best_score']:.4f} "
              f"attempts={s['attempts']} gpu={s['gpu_hours']:.2f}h "
              f"dream_gain={s['dream_gain']:+.4f}")
    print("final policy:", out["policy"]["params"])


if __name__ == "__main__":
    main()
