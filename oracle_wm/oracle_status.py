#!/usr/bin/env python3
"""Quick status check for oracle pipeline runs.

Usage:
    python oracle_wm/oracle_status.py                          # default: oracle_wm/runs
    python oracle_wm/oracle_status.py --run-dir oracle_wm/runs_l3
    python oracle_wm/oracle_status.py --run-dir oracle_wm/runs_l3 --result-dir oracle_results_l3
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Oracle pipeline run status")
    parser.add_argument("--run-dir", default="oracle_wm/runs", help="Directory containing run folders")
    parser.add_argument("--result-dir", default=None, help="Directory containing result folders with summary.json")
    args = parser.parse_args()

    base = Path(args.run_dir)
    result_base = Path(args.result_dir) if args.result_dir else None
    if not base.exists():
        print(f"Not found: {base}")
        return

    runs = sorted(d for d in base.iterdir() if d.is_dir())
    if not runs:
        print(f"No runs in {base}")
        return

    rows = []
    for rd in runs:
        steps = sorted(
            [s for s in rd.iterdir() if s.is_dir() and s.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        committed = sum(1 for s in steps if (s / "selection.json").exists())
        reward = None
        if result_base:
            summary = result_base / rd.name / "summary.json"
            if summary.exists():
                reward = json.loads(summary.read_text()).get("reward")
        rows.append((rd.name, len(steps), committed, reward))

    # print table
    has_rewards = any(r is not None for _, _, _, r in rows)
    hdr = f"{'Task':<75} {'Steps':>5} {'Done':>4}"
    if has_rewards:
        hdr += f" {'Reward':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, ns, nc, r in rows:
        line = f"{name:<75} {ns:>5} {nc:>4}"
        if has_rewards:
            line += f" {r:>6.2f}" if r is not None else "      -"
        print(line)

    print("-" * len(hdr))
    with_steps = sum(1 for _, _, nc, _ in rows if nc > 0)
    rewards = [r for _, _, _, r in rows if r is not None]
    print(f"Total: {len(rows)}  |  With commits: {with_steps}  |  Avg steps: {sum(nc for _,_,nc,_ in rows) / max(len(rows),1):.1f}")
    if rewards:
        print(f"Completed: {len(rewards)}  |  Avg reward: {sum(rewards)/len(rewards):.3f}  |  Success: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")


if __name__ == "__main__":
    main()
