#!/usr/bin/env python3
"""Generate an HTML visualization of an oracle pipeline evaluation run.

Usage:
    python oracle_wm/oracle_html.py --run-dir oracle_wm/runs/<run_dir>
    python oracle_wm/oracle_html.py --task workarena.servicenow.dashboard-... --seed 25
"""

import argparse
import json
import shutil
from pathlib import Path


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 13px;
    background: #13131a;
    color: #d4d4d4;
}

/* ── Top header ─────────────────────────────────────────────────────────── */
.page-header {
    background: #1a1a2e;
    border-bottom: 2px solid #3c3c5c;
    padding: 14px 24px;
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    align-items: baseline;
    gap: 16px;
}
.page-header h1 { font-size: 15px; color: #cdd6f4; font-weight: 600; }
.page-header .meta { color: #666; font-size: 12px; font-family: monospace; }

/* ── Goal box ───────────────────────────────────────────────────────────── */
.goal-box {
    margin: 0;
    padding: 14px 24px;
    background: #181828;
    border-bottom: 1px solid #2a2a3a;
}
.goal-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #5a5a8a;
    margin-bottom: 6px;
}
.goal-text {
    font-size: 13px;
    color: #c0c0d8;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Step nav strip ─────────────────────────────────────────────────────── */
.step-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    padding: 8px 24px;
    background: #16161e;
    border-bottom: 1px solid #2a2a3a;
    position: sticky;
    top: 45px;
    z-index: 99;
}
.step-nav a {
    font-size: 11px;
    font-family: monospace;
    color: #888;
    text-decoration: none;
    padding: 2px 7px;
    border-radius: 3px;
    border: 1px solid #2a2a3a;
    transition: all 0.1s;
}
.step-nav a:hover { color: #cdd6f4; border-color: #5c5c8c; background: #1e1e2e; }

/* ── Step card ──────────────────────────────────────────────────────────── */
.step-card {
    margin: 20px 24px;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    overflow: hidden;
    background: #1a1a26;
}

.step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    background: #20203a;
    border-bottom: 1px solid #2a2a3a;
    flex-wrap: wrap;
}
.step-num { font-size: 13px; font-weight: 700; color: #7f7faf; white-space: nowrap; }
.step-action {
    font-family: monospace;
    font-size: 12px;
    color: #9cdcfe;
    background: #14142a;
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid #3c3c6c;
    word-break: break-all;
}
.bid-tag {
    font-size: 11px;
    font-family: monospace;
    padding: 2px 8px;
    border-radius: 3px;
    white-space: nowrap;
}
.bid-stable     { background: #1a3a1a; color: #6ecf6e; border: 1px solid #2a5a2a; }
.bid-translated { background: #152840; color: #7ecdf7; border: 1px solid #1e4060; }
.bid-other      { background: #2a2a3a; color: #888;    border: 1px solid #3a3a5a; }

/* ── Section label ──────────────────────────────────────────────────────── */
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #5a5a7a;
    padding: 8px 16px 6px;
    background: #16161e;
    border-bottom: 1px solid #222230;
    border-top: 1px solid #222230;
}

/* ── State images (current + decision point) ────────────────────────────── */
.state-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    border-bottom: 1px solid #222230;
}
.state-panel {
    padding: 12px 16px;
    border-right: 1px solid #222230;
}
.state-panel:last-child { border-right: none; }
.state-panel-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #5a5a7a;
    margin-bottom: 8px;
}
.state-panel img {
    width: 100%;
    display: block;
    border-radius: 4px;
    border: 1px solid #2a2a3a;
}
.no-img {
    width: 100%;
    background: #111118;
    border: 1px dashed #2a2a3a;
    border-radius: 4px;
    text-align: center;
    padding: 40px 0;
    color: #333;
    font-size: 12px;
}

/* ── History SOM strip ──────────────────────────────────────────────────── */
.history-strip {
    display: flex;
    gap: 12px;
    overflow-x: auto;
    padding: 12px 16px;
    background: #111118;
    border-bottom: 1px solid #222230;
    scrollbar-width: thin;
    scrollbar-color: #3a3a5a #111118;
}
.history-strip::-webkit-scrollbar { height: 6px; }
.history-strip::-webkit-scrollbar-track { background: #111118; }
.history-strip::-webkit-scrollbar-thumb { background: #3a3a5a; border-radius: 3px; }
.prev-som-item {
    flex: 0 0 200px;
    display: flex;
    flex-direction: column;
    gap: 5px;
}
.prev-som-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #5a5a7a;
    white-space: nowrap;
}
.prev-som-item img {
    width: 100%;
    display: block;
    border-radius: 4px;
    border: 1px solid #2a2a3a;
}

/* ── Candidate tabs ─────────────────────────────────────────────────────── */
.cand-section { border-bottom: 1px solid #222230; }
.cand-tab-bar {
    display: flex;
    gap: 0;
    border-bottom: 1px solid #222230;
    background: #16161e;
    padding: 0 16px;
}
.cand-tab-btn {
    font-size: 12px;
    font-family: monospace;
    padding: 8px 16px;
    cursor: pointer;
    color: #666;
    border: none;
    background: none;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
    white-space: nowrap;
}
.cand-tab-btn:hover { color: #aaa; }
.cand-tab-btn.active { color: #cdd6f4; border-bottom-color: #5c7cfa; }
.cand-tab-btn.chosen { color: #4ec44e; }
.cand-tab-btn.chosen.active { border-bottom-color: #4ec44e; }
.cand-tab-btn.resample { color: #a07040; }
.cand-tab-btn.resample:hover { color: #c89050; }
.cand-tab-btn.resample.active { color: #e6a455; border-bottom-color: #e6a455; }
.cand-tab-btn.resample.chosen { color: #4ec44e; }
.cand-tab-btn.resample.chosen.active { color: #4ec44e; border-bottom-color: #4ec44e; }
.cand-tab-sep {
    font-size: 10px;
    font-family: monospace;
    color: #3a3a5a;
    padding: 8px 10px;
    align-self: center;
    user-select: none;
}

.cand-pane { display: none; padding: 16px; }
.cand-pane.active { display: block; }

.cand-content {
    display: grid;
    grid-template-columns: 50% 1fr;
    gap: 20px;
    align-items: start;
}
.cand-img-wrap img {
    width: 100%;
    display: block;
    border-radius: 4px;
    border: 1px solid #2a2a3a;
}
.cand-meta {
    background: #14141e;
    border: 1px solid #2a2a3a;
    border-radius: 6px;
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.meta-field { display: flex; flex-direction: column; gap: 6px; }
.meta-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #5a5a7a;
}
.meta-value-code {
    font-family: monospace;
    font-size: 15px;
    color: #9cdcfe;
    background: #1a1a2e;
    padding: 6px 10px;
    border-radius: 3px;
    word-break: break-all;
}
.meta-value-text {
    font-size: 14px;
    color: #b0b0c8;
    line-height: 1.6;
    font-style: italic;
}
.meta-value-effect {
    font-size: 13px;
    color: #c8a870;
    line-height: 1.5;
    background: #1e1810;
    border: 1px solid #3a2a10;
    border-radius: 4px;
    padding: 6px 10px;
}
.meta-value-rationale {
    font-size: 14px;
    color: #909090;
    line-height: 1.5;
}
.chosen-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    color: #4ec44e;
    background: #1a3a1a;
    border: 1px solid #2a6a2a;
    padding: 2px 8px;
    border-radius: 4px;
}

/* ── Selection / reasoning ──────────────────────────────────────────────── */
.sel-section {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
}
.sel-panel {
    padding: 14px 16px;
    border-right: 1px solid #222230;
}
.sel-panel:last-child { border-right: none; }
.sel-panel-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #5a5a7a;
    margin-bottom: 10px;
}
.thought-text {
    font-size: 12px;
    color: #b8b8d0;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}
.exec-grid { display: flex; flex-direction: column; gap: 8px; }
.exec-entry { display: flex; flex-direction: column; gap: 3px; }
.exec-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #5a5a7a;
}
.exec-val {
    font-family: monospace;
    font-size: 12px;
    padding: 5px 10px;
    border-radius: 4px;
    word-break: break-all;
}
.exec-val-action  { color: #9cdcfe; background: #14142a; border: 1px solid #2a2a5a; }
.exec-val-exec    { color: #7ecdf7; background: #10202e; border: 1px solid #1e3a52; }
.exec-val-bid     { color: #888;    background: #14141e; border: 1px solid #2a2a3a; }
"""

_JS = """
function showCand(stepId, k) {
    var panes = document.querySelectorAll('#' + stepId + ' .cand-pane');
    var btns  = document.querySelectorAll('#' + stepId + ' .cand-tab-btn');
    for (var i = 0; i < panes.length; i++) {
        panes[i].classList.toggle('active', i === k);
        btns[i].classList.toggle('active', i === k);
    }
}
"""


def _escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _img(path: Path, html_dir: Path, css_class: str = "") -> str:
    if not path.exists():
        return '<div class="no-img">(no image)</div>'
    rel = path.relative_to(html_dir)
    cls = f' class="{css_class}"' if css_class else ""
    return f'<img src="{rel}" loading="lazy"{cls}>'


def _bid_tag(note: str) -> str:
    if not note:
        return ""
    n = note.lower()
    if "stable" in n:
        css, label = "bid-stable", "BID stable"
    elif "translated" in n:
        css, label = "bid-translated", "BID translated"
    else:
        css, label = "bid-other", "BID"
    return f'<span class="bid-tag {css}" title="{_escape(note)}">{label}</span>'


def _attempt_path(step_dir: Path, attempt: str, name: str) -> Path:
    """Resolve a per-attempt file, supporting new (subfoldered) and old (flat) layouts."""
    new = step_dir / attempt / name
    if new.exists():
        return new
    # Old flat layout: resample files had a _resample suffix before the extension
    if attempt == "resample":
        stem, dot, ext = name.rpartition(".")
        return step_dir / (f"{stem}_resample.{ext}" if dot else f"{name}_resample")
    return step_dir / name


def _render_step(step_idx: int, step_dir: Path, html_dir: Path) -> str:
    cands_path    = _attempt_path(step_dir, "initial", "candidates.json")
    rs_cands_path = _attempt_path(step_dir, "resample", "candidates.json")
    sel_path      = step_dir / "selection.json"
    if not cands_path.exists() or not sel_path.exists():
        return ""

    candidates    = json.loads(cands_path.read_text(encoding="utf-8"))
    rs_candidates = json.loads(rs_cands_path.read_text(encoding="utf-8")) if rs_cands_path.exists() else []
    sel           = json.loads(sel_path.read_text(encoding="utf-8"))

    chosen_1      = sel.get("selected_index", 0)   # 1-indexed within its group
    selected_from = sel.get("selected_from", "initial")
    sel_action    = _escape(sel.get("selected_action", ""))
    trans_action  = _escape(sel.get("translated_action", sel_action))
    bid_note      = sel.get("bid_translation", "")
    thought       = _escape(sel.get("thought", ""))

    # Load effects (parallel lists to candidates)
    effects_path    = _attempt_path(step_dir, "initial", "candidate_effects.json")
    rs_effects_path = _attempt_path(step_dir, "resample", "candidate_effects.json")
    effects    = json.loads(effects_path.read_text(encoding="utf-8")) if effects_path.exists() else []
    rs_effects = json.loads(rs_effects_path.read_text(encoding="utf-8")) if rs_effects_path.exists() else []

    step_id = f"step{step_idx}"

    # ── Header ──────────────────────────────────────────────────────────────
    header = f"""
<div class="step-header">
  <span class="step-num">Step {step_idx}</span>
  <span class="step-action">▶ {trans_action}</span>
  {_bid_tag(bid_note)}
</div>"""

    # ── State row: current SOM + decision-point SOM ──────────────────────────
    current_som  = _img(_attempt_path(step_dir, "initial", "current_som.png"), html_dir)
    sel_attempt  = selected_from if selected_from in ("initial", "resample") else "initial"
    dp_som_path  = _attempt_path(step_dir, sel_attempt, "decision_point_som.png")
    decision_som = _img(dp_som_path, html_dir)
    state_row = f"""
<div class="section-label">State Images</div>
<div class="state-row">
  <div class="state-panel">
    <div class="state-panel-label">Current State SOM — at candidate generation</div>
    {current_som}
  </div>
  <div class="state-panel">
    <div class="state-panel-label">Decision-Point SOM — verify BID of executed action</div>
    {decision_som}
  </div>
</div>"""

    # ── History SOM strip (prior steps replayed to verify candidate history) ──
    prev_soms = sorted(
        step_dir.glob("previous_step_*_som.png"),
        key=lambda p: int(p.stem.split("_")[2]),
    )
    history_section = ""
    if prev_soms:
        items = "".join(
            f'<div class="prev-som-item">'
            f'<span class="prev-som-label">Step {p.stem.split("_")[2]} state</span>'
            f'{_img(p, html_dir)}'
            f'</div>'
            for p in prev_soms
        )
        history_section = f"""
<div class="section-label">History Verification — replayed states at each prior step&#39;s decision point</div>
<div class="history-strip">{items}</div>"""

    # ── Candidate tabs (initial C1..CN then resample RS_C1..RS_CM) ───────────
    n_initial = len(candidates)
    if selected_from == "resample":
        chosen_global = n_initial + chosen_1 - 1
    else:
        chosen_global = chosen_1 - 1

    tab_btns = []
    panes    = []

    def _cand_pane(global_k: int, k1: int, cand: dict, effect: str, img_path: Path, is_chosen: bool, label_prefix: str) -> tuple[str, str]:
        is_default = (global_k == chosen_global)
        rs_cls  = " resample" if label_prefix == "RS_" else ""
        btn_cls = "cand-tab-btn" + rs_cls + (" chosen" if is_chosen else "") + (" active" if is_default else "")
        label   = f"{label_prefix}C{k1}" + (" ★" if is_chosen else "")
        btn     = f'<button class="{btn_cls}" onclick="showCand(\'{step_id}\', {global_k})">{label}</button>'

        future_som = _img(img_path, html_dir)
        action_txt = _escape(cand.get("action_text", ""))
        rationale  = _escape(cand.get("rationale", ""))
        action_str = _escape(cand.get("action", ""))
        badge      = '<span class="chosen-badge">CHOSEN</span>' if is_chosen else ""

        effect_html = ""
        if effect:
            effect_html = f"""
      <div class="meta-field">
        <span class="meta-label">Effect</span>
        <span class="meta-value-effect">{_escape(effect)}</span>
      </div>"""

        pane_cls = "cand-pane active" if is_default else "cand-pane"
        pane = f"""
<div class="{pane_cls}">
  <div class="cand-content">
    <div class="cand-img-wrap">{future_som}</div>
    <div class="cand-meta">
      <div class="meta-field">
        <span class="meta-label">{label_prefix}Candidate {k1}</span>
        {badge}
      </div>
      <div class="meta-field">
        <span class="meta-label">Action</span>
        <span class="meta-value-code">{action_str}</span>
      </div>
      <div class="meta-field">
        <span class="meta-label">Description</span>
        <span class="meta-value-text">{action_txt}</span>
      </div>{effect_html}
      <div class="meta-field">
        <span class="meta-label">Rationale</span>
        <span class="meta-value-rationale">{rationale}</span>
      </div>
    </div>
  </div>
</div>"""
        return btn, pane

    for k, cand in enumerate(candidates):
        k1        = k + 1
        is_chosen = (selected_from == "initial" and k1 == chosen_1)
        img_path  = _attempt_path(step_dir, "initial", f"future_{k1}_som.png")
        if not img_path.exists():  # old layout used step-prefixed names
            img_path = step_dir / f"step_{step_idx}_future_{k1}_som.png"
        effect    = effects[k] if k < len(effects) else ""
        btn, pane = _cand_pane(k, k1, cand, effect, img_path, is_chosen, "")
        tab_btns.append(btn)
        panes.append(pane)

    if rs_candidates:
        tab_btns.append('<span class="cand-tab-sep">▸ resample</span>')
        for k, cand in enumerate(rs_candidates):
            k1        = k + 1
            global_k  = n_initial + k
            is_chosen = (selected_from == "resample" and k1 == chosen_1)
            img_path  = _attempt_path(step_dir, "resample", f"future_{k1}_som.png")
            if not img_path.exists():  # old layout used step-prefixed names
                img_path = step_dir / f"step_{step_idx}_future_{k1}_resample_som.png"
            effect    = rs_effects[k] if k < len(rs_effects) else ""
            btn, pane = _cand_pane(global_k, k1, cand, effect, img_path, is_chosen, "RS_")
            tab_btns.append(btn)
            panes.append(pane)

    section_label = "Candidate Future States — toggle to inspect each"
    if rs_candidates:
        section_label += " &nbsp;·&nbsp; <span style='color:#a07040;font-weight:normal;text-transform:none;letter-spacing:0'>resample triggered</span>"

    cand_section = f"""
<div class="section-label">{section_label}</div>
<div class="cand-section">
  <div class="cand-tab-bar">{"".join(tab_btns)}</div>
  {"".join(panes)}
</div>"""

    # ── Selection / reasoning ────────────────────────────────────────────────
    are_same = (sel_action == trans_action)
    exec_label = "Executed Action" if are_same else "Translated (Executed)"
    sel_section = f"""
<div class="section-label">Selection</div>
<div class="sel-section">
  <div class="sel-panel">
    <div class="sel-panel-label">Agent Reasoning</div>
    <div class="thought-text">{thought}</div>
  </div>
  <div class="sel-panel">
    <div class="sel-panel-label">Execution Detail</div>
    <div class="exec-grid">
      <div class="exec-entry">
        <span class="exec-label">Selected Action (original BIDs)</span>
        <span class="exec-val exec-val-action">{sel_action}</span>
      </div>
      <div class="exec-entry">
        <span class="exec-label">{exec_label}</span>
        <span class="exec-val exec-val-exec">{trans_action}</span>
      </div>
      <div class="exec-entry">
        <span class="exec-label">BID Mapping</span>
        <span class="exec-val exec-val-bid">{_escape(bid_note) or "n/a"}</span>
      </div>
    </div>
  </div>
</div>"""

    return f"""
<div class="step-card" id="{step_id}">
  {header}
  {state_row}
  {history_section}
  {cand_section}
  {sel_section}
</div>"""


def _load_goal(run_dir: Path) -> str:
    goal_path = run_dir / "goal.txt"
    if goal_path.exists():
        return goal_path.read_text(encoding="utf-8").strip()
    return ""


def generate_oracle_html(run_dir: Path, output_html: Path | None = None) -> bool:
    """Generate HTML visualization. Returns True if at least one step was rendered."""
    run_dir = Path(run_dir).resolve()
    if output_html is None:
        output_html = run_dir / "oracle_eval.html"
    else:
        output_html = Path(output_html).resolve()

    html_dir = output_html.parent

    step_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not step_dirs:
        return False

    run_name = run_dir.name
    n_steps  = len(step_dirs)
    goal     = _load_goal(run_dir)

    goal_html = ""
    if goal:
        goal_html = f"""
<div class="goal-box">
  <div class="goal-label">Task Goal</div>
  <div class="goal-text">{_escape(goal)}</div>
</div>"""

    nav_links = " ".join(
        f'<a href="#step{int(d.name.split("_")[1])}">{int(d.name.split("_")[1])}</a>'
        for d in step_dirs
    )

    step_blocks = []
    for step_dir in step_dirs:
        step_idx = int(step_dir.name.split("_")[1])
        block = _render_step(step_idx, step_dir, html_dir)
        if block:
            step_blocks.append(block)

    if not step_blocks:
        return False

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Oracle Eval — {_escape(run_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page-header">
  <h1>Oracle Pipeline Evaluation</h1>
  <span class="meta">{_escape(run_name)} &nbsp;·&nbsp; {n_steps} steps</span>
</div>
{goal_html}
<div class="step-nav">{nav_links}</div>
{"".join(step_blocks)}
<script>{_JS}</script>
</body>
</html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return True


def _has_renderable_step(run_dir: Path) -> bool:
    """Check if run_dir has at least one step with candidates.json + selection.json."""
    return any(
        _attempt_path(sd, "initial", "candidates.json").exists() and (sd / "selection.json").exists()
        for sd in run_dir.iterdir() if sd.is_dir() and sd.name.startswith("step_")
    )


def _copy_run_images(run_dir: Path, dest_dir: Path) -> None:
    """Copy step directories (images + json) from run_dir into dest_dir."""
    for item in run_dir.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
        elif item.name == "goal.txt":
            shutil.copy2(item, dest_dir / item.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate oracle pipeline evaluation HTML",
        epilog="Examples:\n"
               "  python oracle_wm/oracle_html.py --out-dir html_output --base-run-dir oracle_wm/runs\n"
               "  python oracle_wm/oracle_html.py --out-dir html_output --run-dir oracle_wm/runs/task_seed0\n"
               "  python oracle_wm/oracle_html.py --out-dir html_output --overwrite\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", required=True,
                        help="Output directory. Each renderable task gets a subfolder with HTML + images.")
    parser.add_argument("--run-dir", help="Path to a specific run directory (skip batch mode)")
    parser.add_argument("--base-run-dir", default="oracle_wm/runs",
                        help="Base directory containing run dirs (default: oracle_wm/runs)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing HTML files")
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Single run mode
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Run directory not found: {run_dir}")
            return
        if not _has_renderable_step(run_dir):
            print(f"No renderable steps in {run_dir}")
            return
        task_dir = out_base / run_dir.name
        task_dir.mkdir(parents=True, exist_ok=True)
        _copy_run_images(run_dir, task_dir)
        generate_oracle_html(task_dir)
        print(f"Created: {task_dir / 'oracle_eval.html'}")
        return

    # Batch mode
    base = Path(args.base_run_dir)
    if not base.exists():
        print(f"Base run directory not found: {base}")
        return

    run_dirs = sorted(d for d in base.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No run directories found in {base}")
        return

    created = skipped = no_steps = 0
    for rd in run_dirs:
        task_dir = out_base / rd.name
        html_out = task_dir / "oracle_eval.html"
        if html_out.exists() and not args.overwrite:
            skipped += 1
            continue
        if not _has_renderable_step(rd):
            no_steps += 1
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        _copy_run_images(rd, task_dir)
        generate_oracle_html(task_dir)
        created += 1

    print(f"Done: {created} created, {skipped} already exist, {no_steps} no renderable steps (skipped)")


if __name__ == "__main__":
    main()
