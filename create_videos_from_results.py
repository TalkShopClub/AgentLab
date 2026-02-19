#!/usr/bin/env python3
"""
Script to process WorkArena L3 task results:
1. Extract goal and actions to JSON
2. Create videos from screenshots with action overlays using ffmpeg
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil
import base64
import gc
import io
from tqdm import tqdm

from browsergym.experiments.loop import ExpResult
from browsergym.experiments import get_exp_result
from agentlab.experiments.loop import EXP_RESULT_CACHE
from PIL import Image, ImageDraw, ImageFont
from agentlab.analyze import inspect_results


def wrap_text(text: str, max_width: int = 80) -> List[str]:
    """Wrap text to fit within max_width characters."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) <= max_width:
            current_line.append(word)
            current_length += len(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def add_text_overlay(image_path: str, output_path: str, action_text: str, step_num: int, goal_text: str = None):
    """Add text overlay to an image using PIL.

    Args:
        image_path: Path to input image
        output_path: Path to save output image
        action_text: Action text to show in bottom left
        step_num: Step number
        goal_text: Goal/task description to show in bottom right (optional)
    """
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Try to use a nice font, fallback to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            font_tiny = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()

        line_height = 18
        padding = 10

        # === LEFT SIDE: Action ===
        max_chars_action = int(img.width / 2 / 9)  # Use half the width
        wrapped_action = wrap_text(action_text, max_chars_action)
        action_height = len(wrapped_action) * line_height + 2 * padding + 30

        # Draw semi-transparent black background for action (left side)
        action_box = [(0, img.height - action_height), (img.width // 2 - 5, img.height)]
        draw.rectangle(action_box, fill=(0, 0, 0, 180))

        # Draw step number
        step_text = f"Step {step_num}"
        draw.text((padding, img.height - action_height + padding),
                 step_text, fill=(255, 255, 0), font=font)

        # Draw action text
        y_offset = img.height - action_height + padding + 25
        for line in wrapped_action:
            draw.text((padding, y_offset), line, fill=(255, 255, 255), font=font_small)
            y_offset += line_height

        # === RIGHT SIDE: Goal/Task Description ===
        if goal_text:
            max_chars_goal = int(img.width / 2 / 9)  # Use half the width
            wrapped_goal = wrap_text(goal_text, max_chars_goal)
            goal_height = len(wrapped_goal) * line_height + 2 * padding + 25

            # Draw semi-transparent black background for goal (right side)
            goal_box = [(img.width // 2 + 5, img.height - goal_height), (img.width, img.height)]
            draw.rectangle(goal_box, fill=(0, 0, 0, 180))

            # Draw "Task:" header
            draw.text((img.width // 2 + 5 + padding, img.height - goal_height + padding),
                     "Task:", fill=(100, 200, 255), font=font)

            # Draw goal text
            y_offset = img.height - goal_height + padding + 20
            for line in wrapped_goal:
                draw.text((img.width // 2 + 5 + padding, y_offset),
                         line, fill=(220, 220, 220), font=font_tiny)
                y_offset += line_height

        img.save(output_path)
        return True
    except Exception as e:
        print(f"Warning: Could not add overlay to {image_path}: {e}")
        # Copy original if overlay fails
        shutil.copy(image_path, output_path)
        return False


def extract_goal_from_axtree(axtree_txt: str) -> str:
    """Extract actual task goal/description from AXTree.

    For WorkArena tasks, the detailed task description is in the Description field
    of the ServiceNow form, which is visible in the AXTree.
    """
    import re

    try:
        lines = axtree_txt.split('\n')

        # Find the Description textbox and extract its content
        for i, line in enumerate(lines):
            if "textbox 'Description'" in line or "textarea 'Description'" in line:
                # The actual text content is in the next few lines as StaticText
                for j in range(i+1, min(i+10, len(lines))):
                    if 'StaticText' in lines[j]:
                        # Extract text between StaticText ' and the last '
                        match = re.search(r"StaticText '(.+)'$", lines[j])
                        if match:
                            description = match.group(1)
                            # Unescape quotes and newlines
                            description = description.replace("\\'", "'")
                            description = description.replace("\\n", "\n")
                            # Return if it looks like a real task description
                            if len(description) > 20:
                                return description

        # If not found in Description field, try Short description
        for i, line in enumerate(lines):
            if "textbox 'Short description'" in line:
                for j in range(i+1, min(i+5, len(lines))):
                    if 'StaticText' in lines[j]:
                        match = re.search(r"StaticText '(.+)'$", lines[j])
                        if match:
                            description = match.group(1)
                            if len(description) > 20:
                                return description

    except Exception as e:
        print(f"Error extracting goal from axtree: {e}")

    return "Goal not available"


def extract_goal_and_actions(exp_dir: str, level: str = 'l3') -> Dict[str, Any]:
    """Extract goal and actions from experiment directory using ExpResult."""
    try:
        import pickle
        import gzip

        exp_result = get_exp_result(exp_dir)

        # Get task metadata from exp_args
        task_name = "Unknown"
        task_seed = None
        if hasattr(exp_result, 'exp_args'):
            exp_args = exp_result.exp_args
            if hasattr(exp_args, 'env_args'):
                task_name = getattr(exp_args.env_args, 'task_name', "Unknown")
                task_seed = getattr(exp_args.env_args, 'task_seed', None)

        # Extract goal - different methods for L2 vs L3
        goal = "Goal not available"
        exp_path = Path(exp_dir)

        if level == 'l2':
            # L2 task: load goal from goal_object.pkl.gz
            goal_file = exp_path / 'goal_object.pkl.gz'
            if goal_file.exists():
                try:
                    with gzip.open(goal_file, 'rb') as f:
                        goal_obj = pickle.load(f)

                    if isinstance(goal_obj, tuple) and len(goal_obj) > 0:
                        if isinstance(goal_obj[0], dict) and 'text' in goal_obj[0]:
                            goal = goal_obj[0]['text']
                except Exception as e:
                    print(f"Warning: Could not load goal from goal_object.pkl.gz: {e}")
        else:
            # L3 task: extract from AXTree
            if len(exp_result.steps_info) > 0:
                first_step = exp_result.steps_info[0]
                if hasattr(first_step, 'obs') and 'axtree_txt' in first_step.obs:
                    goal = extract_goal_from_axtree(first_step.obs['axtree_txt'])

        # Extract all actions and thoughts from steps_info
        actions = []
        for i, step_info in enumerate(exp_result.steps_info):
            action = step_info.action if hasattr(step_info, 'action') else None

            # Extract thought from agent_info
            thought = None
            if hasattr(step_info, 'agent_info') and hasattr(step_info.agent_info, 'think'):
                thought = step_info.agent_info.think

            action_data = {
                "step": i,
                "action": action,
                "thought": thought,
                "reward": step_info.reward if hasattr(step_info, 'reward') else None,
                "terminated": step_info.terminated if hasattr(step_info, 'terminated') else None,
                "truncated": step_info.truncated if hasattr(step_info, 'truncated') else None,
            }

            # Extract world model data from agent_info.extra_info
            if hasattr(step_info, 'agent_info') and hasattr(step_info.agent_info, 'extra_info'):
                extra = step_info.agent_info.extra_info

                if 'wm_candidates' in extra:
                    action_data['wm_candidates'] = extra['wm_candidates']

                if 'wm_predictions' in extra:
                    # Convert numpy arrays to base64 strings for JSON serialization
                    predictions = []
                    for pred in extra['wm_predictions']:
                        pred_copy = pred.copy()
                        if pred_copy.get('image') is not None:
                            import numpy as np
                            img_array = pred_copy['image']
                            img = Image.fromarray(img_array.astype('uint8'))
                            buf = io.BytesIO()
                            img.save(buf, format='PNG')
                            pred_copy['image'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                        predictions.append(pred_copy)
                    action_data['wm_predictions'] = predictions

                if 'wm_mode' in extra:
                    action_data['wm_mode'] = extra['wm_mode']

            actions.append(action_data)

        # Get summary info
        summary_info = exp_result.summary_info if hasattr(exp_result, 'summary_info') else {}

        return {
            "task_name": task_name,
            "task_seed": task_seed,
            "goal": goal,
            "num_steps": len(actions),
            "final_reward": summary_info.get('cum_reward', None),
            "terminated": summary_info.get('terminated', None),
            "truncated": summary_info.get('truncated', None),
            "actions": actions
        }
    except Exception as e:
        print(f"Error extracting data from {exp_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_video_with_overlays(exp_dir: str, task_data: Dict[str, Any], output_video: str) -> bool:
    """Create video from screenshots with action overlays."""
    exp_path = Path(exp_dir)

    # Get all screenshot files
    screenshots = sorted(exp_path.glob("screenshot_step_*.png"),
                        key=lambda x: int(x.stem.split('_')[-1]))

    if not screenshots:
        print(f"No screenshots found in {exp_dir}")
        return False

    # Get the goal text once (will be shown on all frames)
    goal_text = task_data.get('goal', '')

    # Create temporary directory for processed images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Process each screenshot with overlay
        for i, screenshot in enumerate(tqdm(screenshots, desc="  Creating frames", unit="frame", leave=False)):
            # Skip empty files
            if screenshot.stat().st_size == 0:
                continue

            # Get action text for this step
            action_text = "No action"
            if i < len(task_data['actions']):
                action = task_data['actions'][i].get('action', '')
                action_text = f"Action: {action}" if action else "No action"

            # Create image with overlay (now includes goal)
            output_img = temp_path / f"frame_{i:04d}.png"
            add_text_overlay(str(screenshot), str(output_img), action_text, i, goal_text)

        # Get list of processed frames
        frames = sorted(temp_path.glob("frame_*.png"))
        if not frames:
            print(f"No frames created for {exp_dir}")
            return False

        # Create video using ffmpeg
        print(f"Creating video with {len(frames)} frames...")
        frame_pattern = str(temp_path / "frame_%04d.png")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate", "0.5",  # 0.5 frames per second = 2 seconds per frame
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",  # Quality (lower = better, 23 is default)
            output_video
        ]

        try:
            result = subprocess.run(ffmpeg_cmd,
                                   capture_output=True,
                                   text=True,
                                   check=True)
            print(f"Video created: {output_video}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            return False
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg:")
            print("  brew install ffmpeg  # on macOS")
            print("  sudo apt install ffmpeg  # on Ubuntu")
            return False


def process_task_folder(task_dir: str, level: str, overwrite: bool = False) -> bool:
    """Process a single task folder."""
    task_path = Path(task_dir)
    task_name = task_path.name

    # Check if video already exists (previously processed)
    video_path = task_path / "agent_execution.mp4"
    if video_path.exists() and not overwrite:
        print(f"⊘ Skipping {task_name} (already processed)")
        return None

    # Check if summary_info.json exists (indicates task is complete)
    summary_info_path = task_path / "summary_info.json"
    if not summary_info_path.exists():
        print(f"⊘ Skipping {task_name} (no summary_info.json - task incomplete)")
        return None  # Return None to indicate skipped, not failed

    print(f"\n{'='*80}")
    print(f"Processing: {task_name}")
    print(f"{'='*80}")

    # Extract goal and actions
    print("Extracting goal and actions...")
    task_data = extract_goal_and_actions(str(task_path), level)

    if not task_data:
        print(f"Failed to extract data from {task_dir}")
        return False

    # Save to JSON
    json_path = task_path / "task_summary.json"
    with open(json_path, 'w') as f:
        json.dump(task_data, f, indent=2)
    print(f"Saved task summary to: {json_path}")

    # Create video
    video_path = task_path / "agent_execution.mp4"
    success = create_video_with_overlays(str(task_path), task_data, str(video_path))

    if success:
        print(f"✓ Successfully processed {task_name}")
    else:
        print(f"✗ Failed to create video for {task_name}")

    return success


def main():
    """Main function to process all task folders in results directory."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create videos from WorkArena task results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('model_name', type=str, help='Model name to filter')
    parser.add_argument('level', type=str, help='Task level (l2 or l3)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing videos')

    args = parser.parse_args()

    model_name = args.model_name
    level = args.level
    overwrite = args.overwrite

    # Get results directory from AGENTLAB_EXP_ROOT environment variable
    results_base_str = os.environ.get('AGENTLAB_EXP_ROOT', 'results2')
    results_base = Path(results_base_str)

    if not results_base.exists():
        print(f"Error: {results_base} directory not found")
        print(f"AGENTLAB_EXP_ROOT is set to: {results_base_str}")
        sys.exit(1)

    print(f"Using results directory: {results_base}")

    # Get study directories matching the model name and level
    study_dirs = [d for d in results_base.iterdir() if d.is_dir() and model_name in d.name and level in d.name]

    if not study_dirs:
        print(f"No study directories found matching model: {model_name}, level: {level} in {results_base}")
        sys.exit(1)

    if len(study_dirs) > 1:
        print(f"Multiple study directories found matching model: {model_name}, level: {level}:")
        for d in study_dirs:
            print(f"  - {d.name}")
        print("\nUsing the first one...")

    study_dir = study_dirs[0]
    print(f"Processing study: {study_dir.name}\n")

    # Process the study
    total_tasks = 0
    successful_tasks = 0
    skipped_tasks = 0
    failed_tasks = 0

    # Get all task directories in this study
    task_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
    print(f"Found {len(task_dirs)} task directories\n")

    for i, task_dir in enumerate(tqdm(task_dirs, desc="Processing tasks", unit="task")):
        total_tasks += 1
        try:
            result = process_task_folder(str(task_dir), level, overwrite)
            if result is None:
                skipped_tasks += 1
            elif result:
                successful_tasks += 1
            else:
                failed_tasks += 1
            EXP_RESULT_CACHE.pop(str(task_dir), None)
            if i % 20 == 0:
                gc.collect()
        except Exception as e:
            print(f"\nError processing {task_dir.name}: {e}")
            failed_tasks += 1
            continue

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks found: {total_tasks}")
    print(f"Skipped (incomplete): {skipped_tasks}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {failed_tasks}")

    if failed_tasks > 0:
        print("\nNote: Some tasks failed. Check the output above for details.")


if __name__ == "__main__":
    main()
