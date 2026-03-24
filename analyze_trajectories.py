#!/usr/bin/env python3
"""Analyze agent/oracle trajectories for task outcomes and loop detection.

This is a convenience wrapper. The implementation lives in the trajectory_analysis package.

Usage:
    python analyze_trajectories.py <results_dir> [options]
    python -m trajectory_analysis <results_dir> [options]

Options:
    --skip-fingerprint   Use raw action strings instead of fingerprint-normalized
    --skip-llm           Skip LLM-based metrics (no-effect detection)
    --runs-dir DIR       Oracle run artifacts directory (for no-effect detection)
    --output FILE        Write full JSON results to file
"""

from trajectory_analysis.__main__ import main

if __name__ == "__main__":
    main()
