"""CLI entry point for trajectory analysis.

Usage:
    python -m trajectory_analysis <results_dir> [options]
    python analyze_trajectories.py <results_dir> [options]
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .data_loading import detect_pipeline, enumerate_task_dirs, load_reward, load_steps
from .fingerprints import extract_step_data
from .loops import detect_all_loops, compute_looped_steps
from .no_effect import detect_no_effect_steps
from .outcomes import classify_outcome
from .report import print_report
from .types import TaskAnalysis


def analyze_task(
    task_name: str,
    task_dir: Path,
    pipeline: str,
    skip_fingerprint: bool,
    runs_dir: Path | None = None,
    skip_llm: bool = False,
) -> TaskAnalysis:
    """Analyze a single task's trajectory."""
    reward = load_reward(task_dir, pipeline)
    steps = load_steps(task_dir)

    actions, terminated_flags, truncated_flags = extract_step_data(steps, skip_fingerprint)
    outcome = classify_outcome(actions, terminated_flags, truncated_flags, reward)
    loops = detect_all_loops(actions)
    looped_steps, looped_ratio = compute_looped_steps(loops, len(actions))

    # No-effect detection (LLM-based, skippable)
    no_effect = []
    if not skip_llm:
        no_effect = detect_no_effect_steps(task_name, task_dir, pipeline, actions, runs_dir)

    return TaskAnalysis(
        task_name=task_name,
        task_dir=str(task_dir),
        pipeline=pipeline,
        n_steps=len(actions),
        reward=reward,
        outcome=outcome,
        loops=loops,
        has_loops=bool(loops),
        looped_steps=looped_steps,
        looped_step_ratio=round(looped_ratio, 3),
        no_effect_steps=no_effect,
        no_effect_count=len(no_effect),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze agent/oracle trajectories for outcomes and loop detection",
    )
    parser.add_argument("dir", type=str, help="Path to results directory")
    parser.add_argument(
        "--skip-fingerprint",
        action="store_true",
        help="Use raw action strings instead of fingerprint-normalized",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-based metrics (no-effect detection)",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Oracle run artifacts directory (for no-effect detection via candidate_effects)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write full JSON results to file",
    )
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.is_dir():
        print(f"ERROR: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    if runs_dir and not runs_dir.is_dir():
        print(f"ERROR: --runs-dir {runs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    pipeline = detect_pipeline(results_dir)
    task_dirs = enumerate_task_dirs(results_dir, pipeline)
    print(f"Detected pipeline: {pipeline}  |  Found {len(task_dirs)} tasks")
    if runs_dir:
        print(f"Runs dir: {runs_dir}  (no-effect detection enabled)")
    if args.skip_llm:
        print(f"LLM-based metrics: SKIPPED")

    results = []
    for i, (task_name, task_dir) in enumerate(task_dirs):
        short = task_name.split(".")[-1][:50] if "." in task_name else task_name[:50]
        print(f"  Analyzing [{i + 1}/{len(task_dirs)}] {short}...", end="", flush=True)
        try:
            analysis = analyze_task(
                task_name, task_dir, pipeline, args.skip_fingerprint, runs_dir, args.skip_llm
            )
            results.append(analysis)
            status = f"  {analysis.outcome}"
            if analysis.has_loops:
                status += f"  ({len(analysis.loops)} loops)"
            print(status)
        except Exception as e:
            print(f"  ERROR: {e}")

    print_report(results, args.dir)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nFull results written to: {out_path}")


if __name__ == "__main__":
    main()
