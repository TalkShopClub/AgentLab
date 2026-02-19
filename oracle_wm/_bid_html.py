"""Generate a side-by-side HTML comparison of original vs translate-replay trajectories."""

import json
from pathlib import Path


_CSS = """
body { font-family: monospace; font-size: 12px; background: #1e1e1e; color: #d4d4d4; margin: 0; }
h2 { padding: 8px 16px; margin: 0; background: #252526; border-bottom: 1px solid #3c3c3c; }
table { width: 100%; border-collapse: collapse; }
th { background: #2d2d2d; padding: 6px 8px; text-align: left; border-bottom: 2px solid #3c3c3c; position: sticky; top: 0; z-index: 1; }
td { padding: 6px 8px; vertical-align: top; border-bottom: 1px solid #2a2a2a; }
td.step-num { width: 32px; text-align: center; color: #888; font-size: 11px; }
tr.ok td { background: #1e2a1e; }
tr.translated td { background: #1e2227; }
tr.error td { background: #2a1e1e; }
tr.ok:hover td, tr.translated:hover td, tr.error:hover td { filter: brightness(1.15); }
.img-wrap { text-align: center; }
img { max-width: 100%; height: auto; display: block; border: 1px solid #3c3c3c; }
.action { margin-top: 4px; color: #9cdcfe; word-break: break-all; }
.note { margin-top: 2px; color: #888; font-size: 11px; }
.err { margin-top: 2px; color: #f44747; font-size: 11px; white-space: pre-wrap; word-break: break-all; max-height: 80px; overflow: auto; }
.tag { display: inline-block; font-size: 10px; padding: 1px 5px; border-radius: 3px; margin-left: 4px; }
.tag-stable { background: #2d5a2d; color: #9fdf9f; }
.tag-translated { background: #1a3a5a; color: #7ecdf7; }
.tag-ambiguous { background: #4a3a1a; color: #e8c97d; }
.tag-missing { background: #4a2a2a; color: #f47d7d; }
.tag-noerr { background: #2d5a2d; color: #9fdf9f; }
.tag-err { background: #4a2a2a; color: #f47d7d; }
"""

_ROW_TMPL = """
<tr class="{row_class}">
  <td class="step-num">{step}</td>
  <td>
    <div class="img-wrap">{orig_img}</div>
    <div class="action">{orig_action}</div>
  </td>
  <td>
    <div class="img-wrap">{replay_img}</div>
    <div class="action">{translated_action}{translate_tag}</div>
    {note_html}{err_html}
  </td>
</tr>"""


def _img_tag(path: Path, rel_to: Path) -> str:
    if not path.exists():
        return "<span style='color:#666'>(no image)</span>"
    return f'<img src="{path.relative_to(rel_to)}" loading="lazy">'


def _classify_note(note: str) -> str:
    if not note or note == "no-bid-in-action":
        return "noaction"
    if "stable" in note:
        return "stable"
    if "translated" in note:
        return "translated"
    if "ambiguous" in note:
        return "ambiguous"
    if "not found" in note or "not in" in note:
        return "missing"
    return "other"


def generate_comparison_html(json_path: str, output_html: str | None = None) -> None:
    """
    Generate a side-by-side HTML comparison from a translate-replay JSON.
    Left column: original trajectory (from PKL), right column: translated replay.
    """
    json_path = Path(json_path)
    if output_html is None:
        output_html = json_path.parent / (json_path.stem + "_comparison.html")
    else:
        output_html = Path(output_html)

    data = json.loads(json_path.read_text())
    steps = data.get("translated_steps", [])
    task_name = data.get("task_name", "")
    task_seed = data.get("task_seed", "")

    stem = json_path.with_suffix("")
    som_dir = stem / "som"
    som_dir_orig = stem / "som_original"
    html_dir = output_html.parent

    rows = []
    for i, step in enumerate(steps):
        img_name = "step_00_reset.png" if i == 0 else f"step_{i:02d}.png"

        orig_img = _img_tag(som_dir_orig / img_name, html_dir)
        replay_img = _img_tag(som_dir / img_name, html_dir)

        orig_action = step.get("action") or "reset"
        translated_action = step.get("translated_action") or orig_action
        note = step.get("translation_note") or ""
        err = (step.get("action_error") or "").strip()

        note_class = _classify_note(note)
        tag_map = {
            "stable": ("STABLE", "tag-stable"),
            "translated": ("TRANSLATED", "tag-translated"),
            "ambiguous": ("AMBIGUOUS", "tag-ambiguous"),
            "missing": ("MISSING", "tag-missing"),
            "noaction": ("", ""),
            "other": ("", ""),
        }
        tag_label, tag_cls = tag_map.get(note_class, ("", ""))
        translate_tag = f'<span class="tag {tag_cls}">{tag_label}</span>' if tag_label else ""

        note_html = f'<div class="note">{note}</div>' if note else ""
        err_html = f'<div class="err">{err[:300]}</div>' if err else ""

        if err:
            row_class = "error"
        elif note_class == "translated":
            row_class = "translated"
        else:
            row_class = "ok"

        rows.append(_ROW_TMPL.format(
            step=i,
            row_class=row_class,
            orig_img=orig_img,
            replay_img=replay_img,
            orig_action=orig_action,
            translated_action=translated_action,
            translate_tag=translate_tag,
            note_html=note_html,
            err_html=err_html,
        ))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{task_name} seed={task_seed}</title>
<style>{_CSS}</style>
</head>
<body>
<h2>Original vs Translate-Replay &mdash; {task_name} &nbsp; seed={task_seed}</h2>
<table>
<thead><tr>
  <th>#</th>
  <th style="width:47%">Original trajectory</th>
  <th style="width:47%">Translate-replay</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</body>
</html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html)
    print(f"HTML comparison -> {output_html}")
