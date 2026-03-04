#!/usr/bin/env python3
"""Unified HTML visualization for AgentLab (main.py) results.

Replaces the old 3-script pipeline:
  create_videos_from_results.py -> create_html_from_results.py (+ generate_som_screenshots.py)

Reads ExpResult directly from task directories. No intermediate JSON, no video.
Uses existing screenshot_som_step_*.png when --use-som is passed.

Usage:
  python create_html.py -m gpt-5 -l l2
  python create_html.py -m gpt-5 -l l2 --use-som
  python create_html.py -d results2/some_study/
  python create_html.py -t results2/some_study/some_task/
"""

import argparse
import base64
import gc
import gzip
import html as html_mod
import io
import os
import pickle
import re
import sys
from pathlib import Path

from browsergym.experiments import get_exp_result
from agentlab.experiments.loop import EXP_RESULT_CACHE
from PIL import Image

# ---------------------------------------------------------------------------
# CSS (dark theme, adapted from oracle_html.py)
# ---------------------------------------------------------------------------
_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;font-size:13px;background:#13131a;color:#d4d4d4}
.page-header{background:#1a1a2e;border-bottom:2px solid #3c3c5c;padding:14px 24px;position:sticky;top:0;z-index:100;display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
.page-header h1{font-size:15px;color:#cdd6f4;font-weight:600}
.page-header .meta{color:#666;font-size:12px;font-family:monospace}
.goal-box{padding:14px 24px;background:#181828;border-bottom:1px solid #2a2a3a}
.goal-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#5a5a8a;margin-bottom:6px}
.goal-text{font-size:13px;color:#c0c0d8;line-height:1.6;white-space:pre-wrap;word-break:break-word}
.step-nav{display:flex;flex-wrap:wrap;gap:4px;padding:8px 24px;background:#16161e;border-bottom:1px solid #2a2a3a;position:sticky;top:45px;z-index:99}
.step-nav a{font-size:11px;font-family:monospace;color:#888;text-decoration:none;padding:2px 7px;border-radius:3px;border:1px solid #2a2a3a;transition:all .1s}
.step-nav a:hover{color:#cdd6f4;border-color:#5c5c8c;background:#1e1e2e}
.step-card{margin:20px 24px;border:1px solid #2a2a3a;border-radius:8px;overflow:hidden;background:#1a1a26}
.step-header{display:flex;align-items:center;gap:10px;padding:10px 16px;background:#20203a;border-bottom:1px solid #2a2a3a;flex-wrap:wrap}
.step-num{font-size:13px;font-weight:700;color:#7f7faf;white-space:nowrap}
.step-action{font-family:monospace;font-size:12px;color:#9cdcfe;background:#14142a;padding:3px 10px;border-radius:4px;border:1px solid #3c3c6c;word-break:break-all;flex:1;min-width:0}
.state-row{display:grid;grid-template-columns:1fr 1fr;gap:0;border-bottom:1px solid #222230}
.state-panel{padding:12px 16px;border-right:1px solid #222230}
.state-panel:last-child{border-right:none}
.state-panel-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#5a5a7a;margin-bottom:8px}
.state-panel img{width:100%;display:block;border-radius:4px;border:1px solid #2a2a3a}
.no-img{width:100%;background:#111118;border:1px dashed #2a2a3a;border-radius:4px;text-align:center;padding:40px 0;color:#333;font-size:12px}
.info-row{padding:12px 16px;border-bottom:1px solid #222230}
.info-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#5a5a7a;margin-bottom:6px}
.info-text{font-size:12px;color:#b8b8d0;line-height:1.6;white-space:pre-wrap;word-break:break-word}
.info-text-action{color:#9cdcfe;font-family:monospace}
.info-text-thought{color:#c8c090}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:4px;margin-left:10px}
.badge-good{background:#1a3a1a;color:#6ecf6e;border:1px solid #2a5a2a}
.badge-zero{background:#2a2a3a;color:#888;border:1px solid #3a3a5a}
.badge-term{background:#1a3a1a;color:#6ecf6e}
.badge-trunc{background:#3a2a10;color:#c8a870}
.wm-section{padding:12px 16px;border-bottom:1px solid #222230}
.wm-cand{background:#14141e;border:1px solid #2a2a3a;border-radius:6px;padding:12px;margin-bottom:10px}
.wm-cand-header{font-weight:600;color:#5a5a7a;margin-bottom:8px;font-size:11px;text-transform:uppercase}
.wm-field{margin:4px 0;font-size:12px;color:#b0b0c8}
.wm-pred-img{max-width:600px;border:1px solid #2a2a3a;border-radius:4px;margin-top:8px}
"""


def _esc(s):
    return html_mod.escape(str(s or ""))


def _img_tag(path: Path, html_dir: Path) -> str:
    if not path.exists():
        return '<div class="no-img">(no image)</div>'
    return f'<img src="{os.path.relpath(path, html_dir)}" loading="lazy">'


# ---------------------------------------------------------------------------
# Goal extraction
# ---------------------------------------------------------------------------

def _extract_goal_from_axtree(axtree_txt: str) -> str:
    lines = axtree_txt.split("\n")
    for keyword in ("textbox 'Description'", "textarea 'Description'", "textbox 'Short description'"):
        for i, line in enumerate(lines):
            if keyword in line:
                for j in range(i + 1, min(i + 10, len(lines))):
                    m = re.search(r"StaticText '(.+)'$", lines[j])
                    if m and len(m.group(1)) > 20:
                        return m.group(1).replace("\\'", "'").replace("\\n", "\n")
    return ""


def _extract_goal(task_dir: Path, level: str) -> str:
    if level == "l2":
        goal_file = task_dir / "goal_object.pkl.gz"
        if goal_file.exists():
            try:
                with gzip.open(goal_file, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, tuple) and obj and isinstance(obj[0], dict) and "text" in obj[0]:
                    return obj[0]["text"]
            except Exception:
                pass
        return ""
    step0 = task_dir / "step_0.pkl.gz"
    if not step0.exists():
        return ""
    try:
        with gzip.open(step0, "rb") as f:
            info = pickle.load(f)
        axtree = info.obs.get("axtree_txt", "") if hasattr(info, "obs") else ""
        return _extract_goal_from_axtree(axtree)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Per-step rendering
# ---------------------------------------------------------------------------

def _render_step(task_dir: Path, step_num: int, step_info, html_dir: Path, use_som: bool) -> str:
    action = _esc(getattr(step_info, "action", ""))
    thought = ""
    if hasattr(step_info, "agent_info") and hasattr(step_info.agent_info, "think"):
        thought = _esc(step_info.agent_info.think)
    reward = getattr(step_info, "reward", 0)
    terminated = getattr(step_info, "terminated", False)
    truncated = getattr(step_info, "truncated", False)

    # Screenshot paths — use SOM if requested & available, fallback to regular
    prefix = "screenshot_som_step_" if use_som else "screenshot_step_"
    before_path = task_dir / f"{prefix}{step_num}.png"
    after_path = task_dir / f"{prefix}{step_num + 1}.png"
    if use_som and not before_path.exists():
        before_path = task_dir / f"screenshot_step_{step_num}.png"
    if use_som and not after_path.exists():
        after_path = task_dir / f"screenshot_step_{step_num + 1}.png"

    before_img = _img_tag(before_path, html_dir)
    after_img = _img_tag(after_path, html_dir) if after_path.exists() else '<div class="no-img">(final step)</div>'

    badges = ""
    if reward:
        cls = "badge-good" if reward > 0 else "badge-zero"
        badges += f'<span class="badge {cls}">reward={reward:.3f}</span>'
    if terminated:
        badges += '<span class="badge badge-term">terminated</span>'
    elif truncated:
        badges += '<span class="badge badge-trunc">truncated</span>'

    thought_html = ""
    if thought:
        thought_html = f'<div class="info-row"><div class="info-label">Thought</div><div class="info-text info-text-thought">{thought}</div></div>'

    # WM predictions (from agent_info.extra_info, if present)
    wm_html = ""
    if hasattr(step_info, "agent_info") and hasattr(step_info.agent_info, "extra_info"):
        extra = step_info.agent_info.extra_info or {}
        wm_cands = extra.get("wm_candidates", [])
        wm_preds = extra.get("wm_predictions", [])
        wm_mode = extra.get("wm_mode", "text")
        if wm_cands and wm_preds:
            parts = ['<div class="wm-section"><div class="info-label">World Model Predictions</div>']
            for j, (cand, pred) in enumerate(zip(wm_cands, wm_preds), 1):
                parts.append(f'<div class="wm-cand"><div class="wm-cand-header">Candidate #{j}</div>')
                parts.append(f'<div class="wm-field"><strong>Action:</strong> {_esc(cand.get("action", ""))}</div>')
                parts.append(f'<div class="wm-field"><strong>Description:</strong> {_esc(cand.get("action_text", ""))}</div>')
                parts.append(f'<div class="wm-field"><strong>Rationale:</strong> {_esc(cand.get("rationale", ""))}</div>')
                if wm_mode == "image" and pred.get("image") is not None:
                    buf = io.BytesIO()
                    Image.fromarray(pred["image"].astype("uint8")).save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    parts.append(f'<div class="wm-field"><strong>Predicted:</strong><br><img src="data:image/png;base64,{b64}" class="wm-pred-img"></div>')
                elif wm_mode == "text" and pred.get("text"):
                    parts.append(f'<div class="wm-field"><strong>Predicted:</strong> {_esc(pred["text"])}</div>')
                parts.append("</div>")
            parts.append("</div>")
            wm_html = "".join(parts)

    return f"""<div class="step-card" id="step{step_num}">
  <div class="step-header">
    <span class="step-num">Step {step_num}</span>
    <span class="step-action">{action}</span>{badges}
  </div>
  <div class="state-row">
    <div class="state-panel"><div class="state-panel-label">Before Action</div>{before_img}</div>
    <div class="state-panel"><div class="state-panel-label">After Action</div>{after_img}</div>
  </div>
  <div class="info-row"><div class="info-label">Action</div><div class="info-text info-text-action">{action}</div></div>
  {thought_html}{wm_html}
</div>"""


# ---------------------------------------------------------------------------
# Task-level HTML
# ---------------------------------------------------------------------------

def generate_html(task_dir: Path, output_html: Path, use_som: bool, level: str) -> bool:
    html_dir = output_html.parent
    try:
        exp_result = get_exp_result(str(task_dir))
    except Exception as e:
        print(f"  Failed to load: {task_dir.name}: {e}")
        return False

    task_name = "Unknown"
    task_seed = None
    if hasattr(exp_result, "exp_args") and hasattr(exp_result.exp_args, "env_args"):
        task_name = getattr(exp_result.exp_args.env_args, "task_name", "Unknown")
        task_seed = getattr(exp_result.exp_args.env_args, "task_seed", None)

    goal = _extract_goal(task_dir, level)
    n_steps = len(exp_result.steps_info)
    summary = getattr(exp_result, "summary_info", {}) or {}
    final_reward = summary.get("cum_reward")
    reward_str = f"{final_reward:.3f}" if final_reward is not None else "N/A"
    som_label = " (SoM)" if use_som else ""

    goal_html = ""
    if goal:
        goal_html = f'<div class="goal-box"><div class="goal-label">Task Goal</div><div class="goal-text">{_esc(goal)}</div></div>'

    nav = " ".join(f'<a href="#step{i}">{i}</a>' for i in range(n_steps))
    step_blocks = [_render_step(task_dir, i, si, html_dir, use_som) for i, si in enumerate(exp_result.steps_info)]

    page = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{_esc(task_name)}</title>
<style>{_CSS}</style></head>
<body>
<div class="page-header">
  <h1>AgentLab{som_label}: {_esc(task_name)}</h1>
  <span class="meta">seed={task_seed} &middot; {n_steps} steps &middot; reward={reward_str}</span>
</div>
{goal_html}
<div class="step-nav">{nav}</div>
{"".join(step_blocks)}
</body></html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _process_task(task_dir: Path, overwrite: bool, use_som: bool, level: str) -> bool | None:
    """Returns True (created), False (failed), None (skipped)."""
    if not (task_dir / "summary_info.json").exists():
        return None
    suffix = "_som" if use_som else ""
    out = task_dir / f"visualization{suffix}.html"
    if out.exists() and not overwrite:
        return None
    ok = generate_html(task_dir, out, use_som, level)
    if ok:
        print(f"  {task_dir.name} -> {out.name}")
    else:
        print(f"  FAILED: {task_dir.name}")
    return ok


def _process_study(study_dir: Path, overwrite: bool, use_som: bool, level: str):
    task_dirs = sorted(d for d in study_dir.iterdir() if d.is_dir())
    ok = skip = fail = 0
    for i, td in enumerate(task_dirs):
        result = _process_task(td, overwrite, use_som, level)
        if result is None:
            skip += 1
        elif result:
            ok += 1
        else:
            fail += 1
        EXP_RESULT_CACHE.pop(str(td), None)
        if (ok + fail) % 20 == 0:
            gc.collect()
    print(f"  Done: {ok} created, {skip} skipped, {fail} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML visualizations from AgentLab results",
        epilog="Examples:\n"
               "  python create_html.py -m gpt-5 -l l2\n"
               "  python create_html.py -m gpt-5 -l l2 --use-som\n"
               "  python create_html.py -d results2/some_study/\n"
               "  python create_html.py -t results2/some_study/some_task/\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-t", "--task", help="Single task directory")
    parser.add_argument("-d", "--directory", help="Study directory (or parent of study dirs)")
    parser.add_argument("-m", "--model", help="Filter study dirs by model name")
    parser.add_argument("-l", "--level", default="l2", help="Task level for goal extraction & dir filtering (default: l2)")
    parser.add_argument("--use-som", action="store_true", help="Use existing SoM screenshots if available")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing HTML")
    args = parser.parse_args()

    # Single task
    if args.task:
        tp = Path(args.task)
        if not tp.is_dir():
            print(f"Not found: {tp}")
            sys.exit(1)
        result = _process_task(tp, args.overwrite, args.use_som, args.level)
        if result is None:
            print("Skipped (already exists or incomplete)")
        sys.exit(0 if result is not False else 1)

    # Direct directory
    if args.directory:
        d = Path(args.directory)
        if not d.exists():
            print(f"Not found: {d}")
            sys.exit(1)
        has_tasks = any(sub.is_dir() and (sub / "summary_info.json").exists() for sub in d.iterdir() if sub.is_dir())
        if has_tasks:
            print(f"Study: {d.name}")
            _process_study(d, args.overwrite, args.use_som, args.level)
        else:
            for sd in sorted(sd for sd in d.iterdir() if sd.is_dir()):
                print(f"Study: {sd.name}")
                _process_study(sd, args.overwrite, args.use_som, args.level)
        return

    # Auto-discover from AGENTLAB_EXP_ROOT
    results_base = Path(os.environ.get("AGENTLAB_EXP_ROOT", "results2"))
    if not results_base.exists():
        print(f"Not found: {results_base}")
        sys.exit(1)

    study_dirs = [
        d for d in results_base.iterdir() if d.is_dir()
        and (not args.model or args.model in d.name)
        and (not args.level or args.level in d.name)
    ]
    if not study_dirs:
        print(f"No matching study dirs in {results_base}")
        sys.exit(1)

    for sd in sorted(study_dirs):
        print(f"Study: {sd.name}")
        _process_study(sd, args.overwrite, args.use_som, args.level)


if __name__ == "__main__":
    main()
