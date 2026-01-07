#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python calculate_success_rate.py <model_name> [task_level]")
    print("Example: python calculate_success_rate.py gpt-5 l3")
    sys.exit(1)

model_name = sys.argv[1]
task_level = sys.argv[2] if len(sys.argv) > 2 else None

base_path = Path(os.getenv('AGENTLAB_EXP_ROOT'))

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
