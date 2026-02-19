#!/usr/bin/env python3
"""
Extract and save all prompts from completed tasks.
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm
from agentlab.experiments.loop import get_exp_result
import os 

model_name = sys.argv[1] if len(sys.argv) > 1 else 'gpt-5'
level = sys.argv[2] if len(sys.argv) > 2 else 'l2'

results_base = Path(os.getenv("AGENTLAB_EXP_ROOT"))
study_dir = next(d for d in results_base.iterdir() if model_name in d.name and level in d.name)
task_dirs = [d for d in study_dir.iterdir() if d.is_dir() and (d / 'summary_info.json').exists()]

print(f'Processing {len(task_dirs)} tasks...')

for task_dir in tqdm(task_dirs, desc='Saving prompts'):
    try:
        exp = get_exp_result(task_dir)

        all_prompts = []
        for step in exp.steps_info:
            if hasattr(step, 'agent_info') and step.agent_info:
                chat_messages = step.agent_info.get('chat_messages', [])
                if chat_messages:
                    # Convert chat messages to serializable format
                    messages = []
                    for msg in chat_messages:
                        if hasattr(msg, 'to_dict'):
                            messages.append(msg.to_dict())
                        elif isinstance(msg, dict):
                            messages.append(msg)
                        else:
                            messages.append(str(msg))

                    all_prompts.append({
                        'step': step.step,
                        'messages': messages
                    })

        # Save to JSON file in task directory
        output_file = task_dir / 'all_prompts.json'
        with open(output_file, 'w') as f:
            json.dump(all_prompts, f, indent=2)

    except Exception as e:
        print(f'\nError processing {task_dir.name}: {e}')
        continue

print(f'\nDone! Saved prompts for {len(task_dirs)} tasks')
print('Each task folder now contains all_prompts.json')
