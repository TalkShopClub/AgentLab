#!/usr/bin/env python3
"""
BID analysis tool for ServiceNow.

Loads an existing agent trajectory (from the results PKL), replays its actions on
a fresh env (which picks a new instance from the pool), and records BID observations
at each step. Comparing multiple replays reveals whether BIDs are stable across
different pool instances or different loads of the same instance.

Usage (relative --output paths resolve inside oracle_wm/bid_snapshots/):
    # Replay an existing trajectory (original BIDs, new pool instance):
    python oracle_wm/bid_analysis.py replay \\
        results2/<study>/<task_dir> --output replay1.json

    # Replay with BID translation (fingerprint-matched BIDs):
    python oracle_wm/bid_analysis.py translate-replay \\
        results2/<study>/<task_dir> --output translated1.json

    # Compare two replay outputs (or mix original + replay):
    python oracle_wm/bid_analysis.py compare replay1.json replay2.json

    # Compare original vs replayed steps within one replay JSON:
    python oracle_wm/bid_analysis.py self-compare replay1.json

    # Generate side-by-side HTML comparison (output defaults to translated1_comparison.html):
    python oracle_wm/bid_analysis.py html translated1.json
"""

import argparse
from pathlib import Path

from ._bid_compare import compare_original_vs_replay, compare_trajectories
from ._bid_html import generate_comparison_html
from ._bid_replay import replay_trajectory, translate_replay_trajectory
from ._bid_utils import SNAPSHOTS_DIR


def _resolve_output(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = SNAPSHOTS_DIR / p
    return str(p)


def _resolve_input(path: str) -> str:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = SNAPSHOTS_DIR / p
        if candidate.exists():
            return str(candidate)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    rep = sub.add_parser("replay", help="Replay an existing trajectory on a new pool instance")
    rep.add_argument("traj_dir")
    rep.add_argument("--output", required=True)

    trep = sub.add_parser("translate-replay",
                          help="Replay with fingerprint-based BID translation")
    trep.add_argument("traj_dir")
    trep.add_argument("--output", required=True)

    cmp = sub.add_parser("compare", help="Compare two replay JSON files step by step")
    cmp.add_argument("file1")
    cmp.add_argument("file2")

    selfcmp = sub.add_parser("self-compare",
                             help="Compare original vs replayed steps within one replay JSON")
    selfcmp.add_argument("replay_file")

    htm = sub.add_parser("html", help="Generate side-by-side HTML comparison")
    htm.add_argument("json_file")
    htm.add_argument("--output", default=None, help="Output HTML path (default: <json_stem>_comparison.html)")

    args = parser.parse_args()
    if args.mode == "replay":
        replay_trajectory(args.traj_dir, _resolve_output(args.output))
    elif args.mode == "translate-replay":
        translate_replay_trajectory(args.traj_dir, _resolve_output(args.output))
    elif args.mode == "compare":
        compare_trajectories(_resolve_input(args.file1), _resolve_input(args.file2))
    elif args.mode == "html":
        json_file = _resolve_input(args.json_file)
        out_html = _resolve_output(args.output) if args.output else None
        generate_comparison_html(json_file, out_html)
    else:
        compare_original_vs_replay(_resolve_input(args.replay_file))


if __name__ == "__main__":
    main()
