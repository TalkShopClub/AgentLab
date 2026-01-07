#!/usr/bin/env python3
"""Copy HTML files from visual_gpt5_l2 to visual_gpt5_l2_trajectories"""
from pathlib import Path
import shutil
import os 
import argparse

# Create argparser for model name and level
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--level", type=str, required=True)
parser.add_argument("--dir", type=str, required=True)
args = parser.parse_args()

source_dir = Path(args.dir)  
# Source directory is the directory that has model name and level in the name
source_dir = next(d for d in source_dir.iterdir() if args.model in d.name and args.level in d.name)
target_dir = Path(f"{args.model}_{args.level}_trajectories")

# Create target directory if it doesn't exist
target_dir.mkdir(exist_ok=True)

# Iterate through all task folders
for task_folder in source_dir.iterdir():
    if not task_folder.is_dir():
        continue

    # Find all HTML files in task folder
    html_files = list(task_folder.glob("*.html"))

    if html_files:
        # Create corresponding folder in target
        target_task_folder = target_dir / task_folder.name
        target_task_folder.mkdir(exist_ok=True)

        # Copy each HTML file
        for html_file in html_files:
            target_file = target_task_folder / html_file.name
            shutil.copy2(html_file, target_file)
            print(f"Copied {html_file.name} to {target_task_folder.name}")

print(f"\nDone! Copied HTML files from {len(list(target_dir.iterdir()))} task folders")
