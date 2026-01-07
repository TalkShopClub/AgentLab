#!/usr/bin/env python3
import json
import sys
import gc
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from agentlab.experiments.loop import get_exp_result, EXP_RESULT_CACHE

model_name = sys.argv[1] if len(sys.argv) > 1 else 'gpt-5'
level = sys.argv[2] if len(sys.argv) > 2 else 'l2'

results_base = Path('results2')
study_dir = next(d for d in results_base.iterdir() if model_name in d.name and level in d.name)
task_dirs = [d for d in study_dir.iterdir() if d.is_dir() and (d / 'summary_info.json').exists()]

# Load agent flags from first task to determine which fields are visible
first_exp = get_exp_result(task_dirs[0])
obs_flags = first_exp.exp_args.agent_args.flags.obs

# Determine visible fields based on flags
visible_fields = set()
if obs_flags.use_html:
    visible_fields.add(obs_flags.html_type)
if obs_flags.use_ax_tree:
    visible_fields.add('axtree_txt')
visible_fields.update(['goal', 'last_action', 'elapsed_time'])
if obs_flags.use_error_logs:
    visible_fields.add('last_action_error')
if obs_flags.use_focused_element:
    visible_fields.add('focused_element_bid')

print(f'Visible observation fields based on agent config: {sorted(visible_fields)}')

step_context = defaultdict(list)
step_breakdown = defaultdict(lambda: defaultdict(list))
failed_tasks = 0
for i, task_dir in enumerate(tqdm(task_dirs, desc='Analyzing tasks')):
    try:
        exp = get_exp_result(task_dir)
        cumulative_history = 0
        for step in exp.steps_info:
            if step.stats:
                current_obs_tokens = sum(v for k, v in step.stats.items()
                                       if k.startswith('n_token_') and k.replace('n_token_', '') in visible_fields)
                action_tokens = len(str(step.action).split()) if step.action else 0
                thought_tokens = len(str(step.agent_info.get('think', '')).split()) if step.agent_info else 0
                cumulative_history += action_tokens + thought_tokens
                total_context = current_obs_tokens + cumulative_history

                step_context[step.step].append(total_context)
                step_breakdown[step.step]['current_observation'].append(current_obs_tokens)
                step_breakdown[step.step]['cumulative_history'].append(cumulative_history)
            else:
                step_context[step.step].append(0)
                step_breakdown[step.step]['current_observation'].append(0)
                step_breakdown[step.step]['cumulative_history'].append(0)
        EXP_RESULT_CACHE.pop(str(task_dir), None)
        if i % 20 == 0:
            gc.collect()
    except Exception:
        failed_tasks += 1
        continue

avg_data = [{'step': step, 'avg_cumulative': np.mean(contexts), 'n_tasks': len(contexts)}
            for step, contexts in sorted(step_context.items())]

breakdown_data = {}
for step, fields in sorted(step_breakdown.items()):
    breakdown_data[step] = {field: np.mean(tokens) for field, tokens in fields.items()}

with open('context_length.json', 'w') as f:
    json.dump({'summary': avg_data, 'breakdown_by_step': breakdown_data}, f, indent=2)

steps = [d['step'] for d in avg_data]
avg_cumulative = [d['avg_cumulative'] for d in avg_data]
n_tasks = [d['n_tasks'] for d in avg_data]

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(steps, avg_cumulative, alpha=0.7)
ax1.set_xlabel('Step Number')
ax1.set_ylabel('Average Cumulative Context Tokens', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(steps, n_tasks, 'r--', alpha=0.5)
ax2.set_ylabel('Number of Tasks', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title(f'Total Context Length (Current Obs + History) - {model_name} {level.upper()} ({len(task_dirs)} tasks)')
plt.savefig('context_length.png', dpi=150, bbox_inches='tight')
plt.close()

# Breakdown visualization showing current obs vs cumulative history
breakdown_steps = sorted(breakdown_data.keys())
current_obs = [breakdown_data[step].get('current_observation', 0) for step in breakdown_steps]
cumulative_history = [breakdown_data[step].get('cumulative_history', 0) for step in breakdown_steps]

fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(breakdown_steps, current_obs, cumulative_history,
             labels=['Current Observation', 'Cumulative History (Actions + Thoughts)'], alpha=0.8)
ax.set_xlabel('Step Number')
ax.set_ylabel('Tokens')
ax.set_title(f'Context Breakdown: Current Obs vs History - {model_name} {level.upper()}')
ax.legend(loc='upper left')
plt.savefig('context_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'Analyzed {len(task_dirs) - failed_tasks}/{len(task_dirs)} tasks, failed: {failed_tasks}, max step: {max(steps)}')
