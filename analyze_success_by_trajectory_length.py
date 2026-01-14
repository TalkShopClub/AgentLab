#!/usr/bin/env python3
import json
import argparse
import os
from pathlib import Path
from collections import defaultdict


def categorize_by_length(length, threshold1, threshold2):
    """Categorize tasks by trajectory length."""
    if length < threshold1:
        return f"Easy (< {threshold1} steps)"
    elif length <= threshold2:
        return f"Medium ({threshold1}-{threshold2} steps)"
    else:
        return f"Hard (> {threshold2} steps)"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze success rate by oracle trajectory length",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_success_by_trajectory_length.py --json task_results.json --model gpt-5 --level l3
  python analyze_success_by_trajectory_length.py --json task_results.json --model gpt-5 --level l3 --threshold1 10 --threshold2 25
  python analyze_success_by_trajectory_length.py --json task_results.json --model visualagent-gpt-5 --level l2 --dir /path/to/results
        """
    )
    parser.add_argument('--json', type=str, required=True,
                       help='Path to JSON file containing task results')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to filter results')
    parser.add_argument('--level', type=str, required=True,
                       help='Task level to filter (e.g., l2 or l3)')
    parser.add_argument('--dir', type=str,
                       help='Directory containing results (if not specified, uses AGENTLAB_EXP_ROOT)')
    parser.add_argument('--threshold1', type=int, default=20,
                       help='First threshold for categorization (default: 20)')
    parser.add_argument('--threshold2', type=int, default=30,
                       help='Second threshold for categorization (default: 30)')
    parser.add_argument('--use-raw', action='store_true',
                       help='Use raw_trace_length instead of highlevel_trace_length')

    args = parser.parse_args()

    # Load task results
    print(f"Loading task results from {args.json}...")
    with open(args.json, 'r') as f:
        tasks = json.load(f)

    # Determine base path
    base_path = Path(args.dir) if args.dir else Path(os.getenv('AGENTLAB_EXP_ROOT'))

    # Find target folder based on model and level
    matching_folders = [f for f in base_path.iterdir() if f.is_dir() and args.model in f.name and args.level in f.name]

    if not matching_folders:
        print(f"Error: No results folder found matching model={args.model}, level={args.level}")
        return

    results_folder = matching_folders[0]
    print(f"Using results folder: {results_folder.name}")

    # Process each task
    category_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    length_field = 'raw_trace_length' if args.use_raw else 'highlevel_trace_length'

    print(f"Using {length_field} for trajectory length...")

    for task in tasks:
        # Get trajectory length
        trajectory_length = task[length_field]
        task_name = task['task_name']

        # Replace l3 with the specified level if needed
        if args.level and 'l3' in task_name:
            task_name = task_name.replace('l3', args.level)

        # Find matching task folder
        matching_task_folders = list(results_folder.glob(f"*{task_name}*"))

        if not matching_task_folders:
            continue

        task_folder = matching_task_folders[0]
        summary_file = task_folder / 'summary_info.json'

        if not summary_file.exists():
            continue

        # Load success status from summary_info.json
        with open(summary_file) as f:
            summary = json.load(f)

        cum_reward = summary.get('cum_reward', 0)

        # Categorize and accumulate stats
        category = categorize_by_length(trajectory_length, args.threshold1, args.threshold2)
        category_stats[category]['total'] += 1
        category_stats[category]['success'] += cum_reward

    # Calculate and display results
    print("\n" + "="*70)
    print(f"{'Task Splits':<30} {'Ratio (%)':<15} {'Success Rate (%)':<20}")
    print("="*70)

    # Calculate total tasks
    total_tasks = sum(stats['total'] for stats in category_stats.values())

    # Display in order: Easy, Medium, Hard
    categories = [
        f"Easy (< {args.threshold1} steps)",
        f"Medium ({args.threshold1}-{args.threshold2} steps)",
        f"Hard (> {args.threshold2} steps)"
    ]
    for category in categories:
        if category in category_stats:
            stats = category_stats[category]
            ratio = (stats['total'] / total_tasks * 100) if total_tasks > 0 else 0
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0

            print(f"{category:<30} {ratio:<15.1f} {success_rate:<20.1f}")

    print("="*70)
    print(f"Total tasks analyzed: {total_tasks}")


if __name__ == '__main__':
    main()
