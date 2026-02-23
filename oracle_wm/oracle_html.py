#!/usr/bin/env python3
"""Generate an HTML visualization of an oracle pipeline evaluation run.

Usage:
    python oracle_wm/oracle_html.py --debug-dir oracle_wm/debug/<run_dir>
    python oracle_wm/oracle_html.py --task workarena.servicenow.dashboard-... --seed 25
"""

import argparse
import json
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


def _render_step(step_idx: int, step_dir: Path, html_dir: Path) -> str:
    cands_path = step_dir / "candidates.json"
    sel_path   = step_dir / "selection.json"
    if not cands_path.exists() or not sel_path.exists():
        return ""

    candidates = json.loads(cands_path.read_text(encoding="utf-8"))
    sel        = json.loads(sel_path.read_text(encoding="utf-8"))

    chosen_1       = sel.get("selected_index", 0)          # 1-indexed
    sel_action     = _escape(sel.get("selected_action", ""))
    trans_action   = _escape(sel.get("translated_action", sel_action))
    bid_note       = sel.get("bid_translation", "")
    thought        = _escape(sel.get("thought", ""))

    step_id = f"step{step_idx}"

    # ── Header ──────────────────────────────────────────────────────────────
    header = f"""
<div class="step-header">
  <span class="step-num">Step {step_idx}</span>
  <span class="step-action">▶ {trans_action}</span>
  {_bid_tag(bid_note)}
</div>"""

    # ── State row: current SOM + decision-point SOM ──────────────────────────
    current_som  = _img(step_dir / "current_som.png",       html_dir)
    decision_som = _img(step_dir / "decision_point_som.png", html_dir)
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

    # ── Candidate tabs ───────────────────────────────────────────────────────
    tab_btns = []
    panes    = []
    for k, cand in enumerate(candidates):
        k1        = k + 1
        is_chosen = (k1 == chosen_1)
        btn_cls   = "cand-tab-btn chosen active" if (is_chosen) else ("cand-tab-btn" + (" active" if k == 0 and chosen_1 == 0 else ""))
        # Default: show chosen tab; fall back to first
        is_default = is_chosen or (chosen_1 == 0 and k == 0)
        btn_cls = "cand-tab-btn" + (" chosen" if is_chosen else "") + (" active" if is_default else "")

        label = f"C{k1}" + (" ★" if is_chosen else "")
        tab_btns.append(
            f'<button class="{btn_cls}" onclick="showCand(\'{step_id}\', {k})">{label}</button>'
        )

        future_som = _img(step_dir / f"step_{step_idx}_future_{k1}_som.png", html_dir)
        action_txt = _escape(cand.get("action_text", ""))
        rationale  = _escape(cand.get("rationale", ""))
        action_str = _escape(cand.get("action", ""))
        badge      = '<span class="chosen-badge">CHOSEN</span>' if is_chosen else ""

        pane_cls = "cand-pane active" if is_default else "cand-pane"
        panes.append(f"""
<div class="{pane_cls}">
  <div class="cand-content">
    <div class="cand-img-wrap">{future_som}</div>
    <div class="cand-meta">
      <div class="meta-field">
        <span class="meta-label">Candidate {k1}</span>
        {badge}
      </div>
      <div class="meta-field">
        <span class="meta-label">Action</span>
        <span class="meta-value-code">{action_str}</span>
      </div>
      <div class="meta-field">
        <span class="meta-label">Description</span>
        <span class="meta-value-text">{action_txt}</span>
      </div>
      <div class="meta-field">
        <span class="meta-label">Rationale</span>
        <span class="meta-value-rationale">{rationale}</span>
      </div>
    </div>
  </div>
</div>""")

    cand_section = f"""
<div class="section-label">Candidate Future States — toggle to inspect each</div>
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
  {cand_section}
  {sel_section}
</div>"""


def generate_oracle_html(debug_run_dir: Path, output_html: Path | None = None) -> None:
    debug_run_dir = Path(debug_run_dir).resolve()
    if output_html is None:
        output_html = debug_run_dir / "oracle_eval.html"
    else:
        output_html = Path(output_html).resolve()

    html_dir = output_html.parent

    step_dirs = sorted(
        [d for d in debug_run_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not step_dirs:
        print(f"No step_N directories found in {debug_run_dir}")
        return

    run_name = debug_run_dir.name
    n_steps  = len(step_dirs)

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
<div class="step-nav">{nav_links}</div>
{"".join(step_blocks)}
<script>{_JS}</script>
</body>
</html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    print(f"Oracle eval HTML -> {output_html}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate oracle pipeline evaluation HTML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug-dir", help="Path to a specific run debug directory")
    group.add_argument("--task",      help="Task name (used with --seed)")
    parser.add_argument("--seed",          type=int, default=None)
    parser.add_argument("--base-debug-dir", default="oracle_wm/debug")
    parser.add_argument("--output",         default=None)
    args = parser.parse_args()

    if args.debug_dir:
        run_dir = Path(args.debug_dir)
    else:
        if args.seed is None:
            parser.error("--seed is required when using --task")
        run_name = f"{args.task.replace('/', '_')}_seed{args.seed}"
        run_dir  = Path(args.base_debug_dir) / run_name

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    generate_oracle_html(run_dir, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
