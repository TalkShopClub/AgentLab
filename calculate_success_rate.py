#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(
    description="Calculate success rate for AgentLab task results",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python calculate_success_rate.py gpt-5
  python calculate_success_rate.py gpt-5 --level l3
  python calculate_success_rate.py gpt-5 --dir /path/to/results
    """
)
parser.add_argument('--model', type=str, help='Model name to filter results')
parser.add_argument('--level', type=str, help='Task level to filter (e.g., l2 or l3)')
parser.add_argument('--dir', type=str, help='Directory containing results (if not specified, uses AGENTLAB_EXP_ROOT)')

args = parser.parse_args()

model_name = args.model
task_level = args.level

base_path = Path(args.dir) if args.dir else Path(os.getenv('AGENTLAB_EXP_ROOT'))

if task_level:
    target_folder = next(f for f in base_path.iterdir() if model_name in f.name and task_level in f.name)
else:
    target_folder = next(f for f in base_path.iterdir() if model_name in f.name)

total_reward = sum(
    json.load(open(task / 'summary_info.json')).get('cum_reward', 0)
    for task in target_folder.iterdir()
    if task.is_dir() and (task / 'summary_info.json').exists()
)

task_count = sum(1 for task in target_folder.iterdir() if task.is_dir() and (task / 'summary_info.json').exists())

print(f'Tasks: {task_count}, Reward: {total_reward}, Success: {total_reward/task_count:.2%}')
