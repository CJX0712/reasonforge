"""
Benchmark sweep — accuracy vs compute budget, averaged over seeds.

Usage::

    python examples/bench_sweep.py
    python examples/bench_sweep.py --budgets 4,8,16,32 --seeds 5

Writes ``results/sweep.json`` and prints a markdown table.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasonforge import ReasonForge  # noqa: E402
from reasonforge.bench.tasks import TASKS  # noqa: E402
from reasonforge.core.budget import BudgetConfig  # noqa: E402
from reasonforge.llm.mock import MockStudent  # noqa: E402

STRATS = ["greedy", "self_consistency", "best_of_n_verified", "vwsc", "vwsc_adaptive", "dvts"]


def run(budgets: List[int], n_seeds: int, error_rate: float) -> Dict[str, object]:
    out: Dict[str, object] = {"budgets": budgets, "seeds": n_seeds, "error_rate": error_rate, "rows": []}
    for budget in budgets:
        row: Dict[str, object] = {"budget": budget, "strategies": {}}
        for name in STRATS:
            accs: List[float] = []
            samples: List[float] = []
            eces: List[float] = []
            for s in range(n_seeds):
                rf = ReasonForge(
                    backend=MockStudent(error_rate=error_rate),
                    seed=s,
                    budget=BudgetConfig(max_traces=budget, initial_batch=max(2, budget // 2)),
                )
                rep = rf.benchmark(strategies=[name], tasks=TASKS, salt=s * 101)
                r = rep[name]
                accs.append(r.accuracy)
                samples.append(r.mean_samples)
                eces.append(r.calibration()["ece"])
            row["strategies"][name] = {  # type: ignore[index]
                "accuracy": statistics.mean(accs),
                "accuracy_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "samples": statistics.mean(samples),
                "ece": statistics.mean(eces),
            }
        out["rows"].append(row)  # type: ignore[union-attr]
    return out


def to_markdown(result: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append(f"Simulated student error rate: {result['error_rate']} | seeds per cell: {result['seeds']}")
    lines.append("")
    for row in result["rows"]:  # type: ignore[union-attr]
        b = row["budget"]  # type: ignore[index]
        lines.append(f"### Compute budget = {b} samples (max)")
        lines.append("")
        lines.append("| strategy | accuracy | std | samples used | ECE |")
        lines.append("|---|---|---|---|---|")
        for name, m in sorted(row["strategies"].items(), key=lambda kv: -kv[1]["accuracy"]):  # type: ignore[union-attr]
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.1f} | {:.3f} |".format(
                    name, m["accuracy"], m["accuracy_std"], m["samples"], m["ece"]
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="4,8,16,32")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--error-rate", type=float, default=0.45)
    ap.add_argument("--out", default="results/sweep.json")
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    result = run(budgets, args.seeds, args.error_rate)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(to_markdown(result))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
