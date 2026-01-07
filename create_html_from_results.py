#!/usr/bin/env python3
"""
Script to create HTML visualizations from AgentLab task results.
Run this after create_videos_from_results.py to generate action screenshot HTML files.
"""

import json
import os
import sys
import base64
import argparse
from pathlib import Path
from typing import Dict, Any


def create_html_visualization(exp_dir: str, task_data: Dict[str, Any], output_html: str, use_som: bool = False) -> bool:
    """Create HTML visualization of action screenshots similar to UI-Cube format."""
    exp_path = Path(exp_dir)

    # Choose screenshot pattern based on use_som flag
    screenshot_pattern = "screenshot_som_step_*.png" if use_som else "screenshot_step_*.png"
    screenshots = sorted(exp_path.glob(screenshot_pattern),
                        key=lambda x: int(x.stem.split('_')[-1]))

    if not screenshots:
        return False

    html_parts = []
    screenshot_type = "SoM Screenshots" if use_som else "Action Screenshots"
    html_parts.append(("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{screenshot_type}: {task_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .task-info {{
            color: #666;
            font-size: 14px;
            margin: 5px 0;
        }}
        .goal {{
            background: #f8f9fa;
            padding: 15px;
            margin-top: 15px;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }}
        .goal strong {{
            color: #007bff;
        }}
        .step {{
            background: white;
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .step-header {{
            background: #007bff;
            color: white;
            padding: 12px 20px;
            font-weight: 600;
        }}
        .step-content {{
            padding: 20px;
        }}
        .screenshot {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .screenshots-container {{
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
        }}
        .screenshot-box {{
            flex: 1;
        }}
        .screenshot-label {{
            font-size: 12px;
            font-weight: 600;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}
        .action-info {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        .action-info strong {{
            color: #28a745;
        }}
        .thought-info {{
            background: #fff3cd;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
            border-left: 4px solid #ffc107;
        }}
        .thought-info strong {{
            color: #856404;
        }}
        .meta-info {{
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }}
        .reward {{
            display: inline-block;
            padding: 4px 8px;
            background: #e7f3ff;
            border-radius: 4px;
            margin-right: 10px;
        }}
        .status {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .status.terminated {{
            background: #d4edda;
            color: #155724;
        }}
        .status.truncated {{
            background: #fff3cd;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{screenshot_type}: {task_name}</h1>
        <div class="task-info">Task Seed: {task_seed}</div>
        <div class="task-info">Total Steps: {num_steps}</div>
        <div class="task-info">Final Reward: {final_reward}</div>
        <div class="goal">
            <strong>Task Goal:</strong><br>
            {goal}
        </div>
    </div>
""").format(
        screenshot_type=screenshot_type,
        task_name=task_data.get('task_name', 'Unknown'),
        task_seed=task_data.get('task_seed', 'N/A'),
        num_steps=task_data.get('num_steps', 0),
        final_reward=task_data.get('final_reward', 'N/A'),
        goal=task_data.get('goal', 'Goal not available').replace('\n', '<br>')
    ))

    for i, screenshot in enumerate(screenshots):
        if screenshot.stat().st_size == 0:
            continue

        with open(screenshot, 'rb') as img_file:
            img_data_before = base64.b64encode(img_file.read()).decode('utf-8')

        # Get next screenshot for "after" view
        img_data_after = None
        if i + 1 < len(screenshots) and screenshots[i + 1].stat().st_size > 0:
            with open(screenshots[i + 1], 'rb') as img_file:
                img_data_after = base64.b64encode(img_file.read()).decode('utf-8')

        action_data = task_data['actions'][i] if i < len(task_data['actions']) else {}
        action = action_data.get('action', 'No action')
        thought = action_data.get('thought', '')
        reward = action_data.get('reward', 0)
        terminated = action_data.get('terminated', False)
        truncated = action_data.get('truncated', False)

        if action:
            action = str(action).replace('\n', '<br>')
        if thought:
            thought = str(thought).replace('\n', '<br>')

        status_html = ''
        if terminated:
            status_html = '<span class="status terminated">Terminated</span>'
        elif truncated:
            status_html = '<span class="status truncated">Truncated</span>'

        thought_html = ''
        if thought:
            thought_html = f"""
            <div class="thought-info">
                <strong>Thought:</strong><br>
                {thought}
            </div>"""

        # Build screenshots HTML
        if img_data_after:
            screenshots_html = f"""
            <div class="screenshots-container">
                <div class="screenshot-box">
                    <div class="screenshot-label">Before Action</div>
                    <img src="data:image/png;base64,{img_data_before}" alt="Before step {i}" class="screenshot">
                </div>
                <div class="screenshot-box">
                    <div class="screenshot-label">After Action</div>
                    <img src="data:image/png;base64,{img_data_after}" alt="After step {i}" class="screenshot">
                </div>
            </div>"""
        else:
            screenshots_html = f"""
            <div class="screenshots-container">
                <div class="screenshot-box">
                    <div class="screenshot-label">Screenshot</div>
                    <img src="data:image/png;base64,{img_data_before}" alt="Screenshot step {i}" class="screenshot">
                </div>
            </div>"""

        html_parts.append(f"""
    <div class="step">
        <div class="step-header">Step {i}</div>
        <div class="step-content">
            {screenshots_html}
            <div class="action-info">
                <strong>Action:</strong><br>
                {action}
            </div>{thought_html}
            <div class="meta-info">
                <span class="reward">Reward: {reward}</span>
                {status_html}
            </div>
        </div>
    </div>
""")

    html_parts.append("""
</body>
</html>""")

    try:
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(''.join(html_parts))
        return True
    except Exception as e:
        print(f"Error creating HTML: {e}")
        return False


def process_task_folder(task_dir: Path, overwrite: bool = False, use_som: bool = False) -> bool:
    """Process a single task folder to create HTML visualization."""
    task_name = task_dir.name

    # Use different filename based on use_som flag
    html_filename = "action_screenshots_som.html" if use_som else "action_screenshots.html"
    html_path = task_dir / html_filename
    if html_path.exists() and not overwrite:
        print(f"⊘ Skipping {task_name} (already processed)")
        return None

    summary_path = task_dir / "task_summary.json"
    if not summary_path.exists():
        print(f"⊘ Skipping {task_name} (no task_summary.json - run create_videos_from_results.py first)")
        return None

    with open(summary_path, 'r') as f:
        task_data = json.load(f)

    success = create_html_visualization(str(task_dir), task_data, str(html_path), use_som=use_som)

    if success:
        print(f"✓ Created HTML for {task_name}")
    else:
        print(f"✗ Failed to create HTML for {task_name}")

    return success


def main():
    """Main function to create HTML visualizations for all tasks."""

    parser = argparse.ArgumentParser(
        description="Create HTML visualizations from AgentLab task results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_html_from_results.py --model gpt-5
  python create_html_from_results.py -m claude-sonnet -l l3
  python create_html_from_results.py -d visual_gpt5_l2 --use-som
  python create_html_from_results.py  # Process all models
        """
    )
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default=None,
        help='Directory containing task folders (if not specified, uses AGENTLAB_EXP_ROOT and filters by model/level)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        help='Model name to filter study directories (only used if --directory is not specified)'
    )
    parser.add_argument(
        '-l', '--level',
        type=str,
        default=None,
        help='Task level to filter (e.g., l2 or l3) (only used if --directory is not specified)'
    )
    parser.add_argument(
        '--use-som',
        action='store_true',
        help='Use Set-of-Marks (SoM) screenshots instead of regular screenshots'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing HTML files'
    )

    args = parser.parse_args()

    # Handle direct directory specification
    if args.directory:
        results_base = Path(args.directory)
    else:
        results_base_str = os.environ.get('AGENTLAB_EXP_ROOT', 'results2')
        results_base = Path(results_base_str)

    if not results_base.exists():
        print(f"Error: {results_base} directory not found")
        if not args.directory:
            print(f"AGENTLAB_EXP_ROOT is set to: {results_base_str}")
        sys.exit(1)

    print(f"Using results directory: {results_base}")

    # If directory is specified directly, treat it as containing task folders
    if args.directory:
        # Check if this directory contains task folders (has step_*.pkl.gz files)
        has_task_folders = any(d.is_dir() and list(d.glob("step_*.pkl.gz")) for d in results_base.iterdir() if d.is_dir())

        if has_task_folders:
            # Directory contains task folders, treat it as a single study
            study_dirs = [results_base]
        else:
            # Directory might contain study directories
            study_dirs = [d for d in results_base.iterdir() if d.is_dir()]
            if not study_dirs:
                print(f"No task or study directories found in {results_base}")
                sys.exit(1)
    elif args.model or args.level:
        filters = []
        if args.model:
            filters.append(f"model: {args.model}")
        if args.level:
            filters.append(f"level: {args.level}")

        study_dirs = [d for d in results_base.iterdir() if d.is_dir()
                     and (not args.model or args.model in d.name)
                     and (not args.level or args.level in d.name)]

        if not study_dirs:
            print(f"No study directories found matching {', '.join(filters)} in {results_base}")
            sys.exit(1)
        print(f"Filtering for {', '.join(filters)}")
    else:
        study_dirs = [d for d in results_base.iterdir() if d.is_dir()]
        if not study_dirs:
            print(f"No study directories found in {results_base}")
            sys.exit(1)
        print("Processing all study directories")

    if len(study_dirs) > 1 and (args.model or args.level):
        print(f"Multiple study directories found:")
        for d in study_dirs:
            print(f"  - {d.name}")
        print("\nUsing the first one...")
        study_dirs = [study_dirs[0]]

    print(f"\nProcessing {len(study_dirs)} study director{'y' if len(study_dirs) == 1 else 'ies'}\n")

    total_tasks = 0
    successful_tasks = 0
    skipped_tasks = 0
    failed_tasks = 0

    for study_dir in study_dirs:
        print(f"{'='*80}")
        print(f"Study: {study_dir.name}")
        print(f"{'='*80}")

        task_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
        print(f"Found {len(task_dirs)} task directories\n")

        for task_dir in task_dirs:
            total_tasks += 1
            result = process_task_folder(task_dir, overwrite=args.overwrite, use_som=args.use_som)
            if result is None:
                skipped_tasks += 1
            elif result:
                successful_tasks += 1
            else:
                failed_tasks += 1

        print()

    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks found: {total_tasks}")
    print(f"Skipped (no summary): {skipped_tasks}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {failed_tasks}")


if __name__ == "__main__":
    main()
