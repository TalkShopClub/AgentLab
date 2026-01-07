#!/usr/bin/env python3
"""
Generate Set-of-Mark (SoM) screenshots from existing experiment data.

This script reads the stored step data and generates SoM screenshots with
bounding boxes and bid labels overlaid on the original screenshots.
"""
import pickle
import gzip
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Import the overlay_som function from browsergym
from browsergym.utils.obs import overlay_som


def generate_som_for_task(task_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    """Generate SoM screenshots for a single task directory.

    Args:
        task_dir: Path to task directory
        overwrite: Whether to overwrite existing SoM screenshots

    Returns:
        Tuple of (successful_count, failed_count)
    """
    success_count = 0
    failed_count = 0

    # Get all step pickle files
    step_files = sorted(task_dir.glob("step_*.pkl.gz"),
                       key=lambda x: int(x.stem.replace('.pkl', '').split('_')[1]))

    for step_file in step_files:
        step_num = int(step_file.stem.replace('.pkl', '').split('_')[1])
        som_screenshot_path = task_dir / f"screenshot_som_step_{step_num}.png"

        # Skip if already exists and not overwriting
        if som_screenshot_path.exists() and not overwrite:
            success_count += 1
            continue

        try:
            # Load step data
            with gzip.open(step_file, 'rb') as f:
                step_data = pickle.load(f)

            # Check if we have the required data
            if not hasattr(step_data, 'obs') or not isinstance(step_data.obs, dict):
                failed_count += 1
                continue

            obs = step_data.obs

            if 'extra_element_properties' not in obs:
                failed_count += 1
                continue

            # Get the regular screenshot
            screenshot_path = task_dir / f"screenshot_step_{step_num}.png"
            if not screenshot_path.exists():
                failed_count += 1
                continue

            # Load screenshot
            screenshot = np.array(Image.open(screenshot_path))
            screenshot_height, screenshot_width = screenshot.shape[:2]

            # Calculate scale factor by finding the first full-viewport element
            # (usually element 0) and comparing its bbox to screenshot dimensions
            scale_x = scale_y = 1.0
            for bid, props in obs['extra_element_properties'].items():
                if props.get('bbox') is not None:
                    x, y, w, h = props['bbox']
                    # If this element spans close to the full viewport, use it to calculate scale
                    if w > screenshot_width and h > screenshot_height:
                        scale_x = screenshot_width / w
                        scale_y = screenshot_height / h
                        break

            # Scale bounding boxes to match screenshot resolution
            scaled_props = {}
            for bid, props in obs['extra_element_properties'].items():
                scaled_props[bid] = props.copy()
                if props.get('bbox') is not None:
                    x, y, w, h = props['bbox']
                    # Apply scaling
                    scaled_props[bid]['bbox'] = [
                        x * scale_x,
                        y * scale_y,
                        w * scale_x,
                        h * scale_y
                    ]

            # Generate SoM screenshot using overlay_som with scaled coordinates
            screenshot_som = overlay_som(
                screenshot,
                extra_properties=scaled_props
            )

            # Save SoM screenshot
            img = Image.fromarray(screenshot_som)
            img.save(som_screenshot_path)

            success_count += 1

        except Exception as e:
            print(f"  Error processing {step_file.name}: {e}")
            failed_count += 1
            continue

    return success_count, failed_count


def main():
    """Main function to generate SoM screenshots for all tasks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Set-of-Mark (SoM) screenshots from stored experiment data"
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Folder containing task directories (e.g., visual_gpt5_l2)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing SoM screenshots'
    )

    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder {folder} does not exist")
        sys.exit(1)

    # Get all task directories (directories with step pickle files)
    task_dirs = [d for d in folder.iterdir()
                 if d.is_dir() and list(d.glob("step_*.pkl.gz"))]

    if not task_dirs:
        print(f"No task directories found in {folder}")
        sys.exit(1)

    print(f"Found {len(task_dirs)} task directories")
    print(f"Generating SoM screenshots...\n")

    total_success = 0
    total_failed = 0
    total_tasks_processed = 0

    for task_dir in tqdm(task_dirs, desc="Processing tasks", unit="task"):
        success, failed = generate_som_for_task(task_dir, args.overwrite)
        total_success += success
        total_failed += failed
        if success + failed > 0:
            total_tasks_processed += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Tasks processed: {total_tasks_processed}")
    print(f"SoM screenshots generated: {total_success}")
    print(f"Failed: {total_failed}")

    if total_failed > 0:
        print("\nNote: Some screenshots failed to generate. Check the output above for details.")


if __name__ == "__main__":
    main()
