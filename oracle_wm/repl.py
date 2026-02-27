#!/usr/bin/env python3
"""Interactive BrowsingGym REPL for manual action testing.

Fires up a ServiceNow instance, resets to the given task/seed, then enters a
read-eval-print loop: type an action at the prompt, it executes in Playwright,
and the AXTree + SOM are immediately overwritten to --out-dir so you can watch
them update in your editor/viewer side by side.

Usage:
    python oracle_wm/repl.py --task workarena.servicenow.basic-expense-management-medium-l2 --seed 30
    python oracle_wm/repl.py --task workarena.servicenow.basic-expense-management-medium-l2 --seed 30 --out-dir /tmp/repl
"""

import argparse
import copy
import datetime
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.wm_visual_agent.agent_configs import DEFAULT_OBS_FLAGS
from agentlab.experiments.loop import EnvArgs

from agentlab.utils.phantom_actions import _resolve_clickable_bbox, resolve_phantom_action
from oracle_wm._bid_utils import _make_env, _wait_idle, get_valid_snow_instance


def _draw_bbox_and_center(draw: ImageDraw.ImageDraw, bbox: dict, color, r: int = 10) -> None:
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
    cx, cy = x + w / 2, y + h / 2
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=color, width=2)
    draw.line([(cx - r * 2, cy), (cx + r * 2, cy)], fill=color, width=2)
    draw.line([(cx, cy - r * 2), (cx, cy + r * 2)], fill=color, width=2)


def _cmd_coord(bid: str, env, obs: dict, out_dir: Path) -> None:
    """Print bbox info for a BID, run the phantom walk, and save an annotated screenshot.

    Raw bbox + center always drawn in red. If the element is a phantom, the resolved
    ancestor bbox + center is additionally drawn in green.
    """
    from browsergym.core.action.utils import get_elem_by_bid
    from agentlab.utils.phantom_actions import _PHANTOM_MIN_AREA
    page = env.unwrapped.page
    try:
        elem = get_elem_by_bid(page, bid)
        raw_bbox = elem.bounding_box()
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    if raw_bbox is None:
        print(f"  {bid!r}: no bounding box (not visible or zero-size)")
        return

    raw_area = raw_bbox["width"] * raw_bbox["height"]
    print(f"  {bid}: bbox=({raw_bbox['x']:.1f}, {raw_bbox['y']:.1f}, w={raw_bbox['width']:.1f}, h={raw_bbox['height']:.1f})  area={raw_area:.0f}")

    resolved_bbox, depth, parent_bid = _resolve_clickable_bbox(elem)

    if raw_area <= _PHANTOM_MIN_AREA:
        if resolved_bbox is not None:
            rcx = resolved_bbox["x"] + resolved_bbox["width"] / 2
            rcy = resolved_bbox["y"] + resolved_bbox["height"] / 2
            bid_str = f"  parent_bid={parent_bid!r}" if parent_bid else ""
            print(f"  PHANTOM — resolved to DOM ancestor depth={depth}{bid_str}: "
                  f"bbox=({resolved_bbox['x']:.1f}, {resolved_bbox['y']:.1f}, "
                  f"w={resolved_bbox['width']:.1f}, h={resolved_bbox['height']:.1f})  "
                  f"center=({rcx:.1f}, {rcy:.1f})")
        else:
            print(f"  PHANTOM — no suitable ancestor found")
    else:
        cx = raw_bbox["x"] + raw_bbox["width"] / 2
        cy = raw_bbox["y"] + raw_bbox["height"] / 2
        print(f"  center=({cx:.1f}, {cy:.1f})")

    sc = obs.get("screenshot")
    if sc is None:
        return
    img = Image.fromarray(sc).copy()
    draw = ImageDraw.Draw(img)

    # Always draw raw bbox + center in red
    _draw_bbox_and_center(draw, raw_bbox, color=(255, 0, 0))

    # If phantom, also draw resolved ancestor bbox + center in green
    if raw_area <= _PHANTOM_MIN_AREA and resolved_bbox is not None:
        _draw_bbox_and_center(draw, resolved_bbox, color=(0, 180, 80))

    out_path = out_dir / f"coord_{bid}.png"
    img.save(out_path)
    print(f"  Saved: {out_path}")


def _save_state(obs: dict, out_dir: Path, step: int, action: str, reward: float, error: str) -> None:
    axtree = obs.get("axtree_txt", "")
    (out_dir / "axtree.txt").write_text(axtree, encoding="utf-8")

    sc = obs.get("screenshot")
    if sc is not None:
        Image.fromarray(sc).save(out_dir / "screenshot.png")

    sc_som = obs.get("screenshot_som")
    if sc_som is not None:
        Image.fromarray(sc_som).save(out_dir / "som.png")

    with open(out_dir / "log.txt", "a", encoding="utf-8") as f:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        f.write(f"[{ts}] step={step}  action={action!r}  reward={reward:.3f}  error={error!r}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive BrowsingGym REPL")
    parser.add_argument("--task",     required=True, help="BrowserGym task name")
    parser.add_argument("--seed",     type=int, required=True, help="Task seed")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--out-dir",  default="oracle_wm/repl_out", help="Directory for output files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    obs_flags = copy.deepcopy(DEFAULT_OBS_FLAGS)
    obs_flags.use_ax_tree = True
    obs_preprocessor = dp.make_obs_preprocessor(obs_flags)

    instance = get_valid_snow_instance()
    env_args = EnvArgs(task_name=args.task, task_seed=args.seed, headless=args.headless)
    env = _make_env(env_args, instance)

    try:
        obs, _ = env.reset(seed=args.seed)
        _wait_idle(env)
        obs = obs_preprocessor(obs)
        _save_state(obs, out_dir, step=0, action="<reset>", reward=0.0, error="")

        goal = obs.get("goal", "")
        print(f"\nGoal: {goal}")
        print(f"\nOutputs updating in: {out_dir.resolve()}/")
        print("  axtree.txt    — AXTree with BIDs (overwritten each step)")
        print("  som.png       — Set-of-Marks screenshot (overwritten each step)")
        print("  screenshot.png — Plain screenshot (overwritten each step)")
        print("  log.txt       — append-only action log")
        print("\nActions: click('a60')  press('a60', 'Space')  fill('a73', 'text')  select_option('a60', 'Short description')")
        print("Special:  coord('a60')  — print bbox center + save annotated screenshot (no env step)")
        print("Type 'quit' to exit.\n")

        step = 1
        while True:
            try:
                action = input(f"[step {step}] >>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if action.lower() in ("quit", "exit", "q"):
                break
            if not action:
                continue

            m = re.match(r"""^coord\(['"]([^'"]+)['"]\)$""", action)
            if m:
                _cmd_coord(m.group(1), env, obs, out_dir)
                continue

            resolved = resolve_phantom_action(action, env)
            if resolved != action:
                print(f"  phantom resolved: {action!r} -> {resolved!r}")
            obs, reward, terminated, truncated, _ = env.step(resolved)
            _wait_idle(env)
            obs = obs_preprocessor(obs)

            error = obs.get("last_action_error") or ""
            _save_state(obs, out_dir, step=step, action=action, reward=reward, error=error)

            print(f"  reward={reward:.3f}  terminated={terminated}  truncated={truncated}")
            if error:
                print(f"  ERROR: {error}")

            step += 1

            if terminated or truncated:
                print("Episode ended.")
                break

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
