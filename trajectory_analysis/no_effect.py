"""LLM-based no-effect (grounding failure) detection for oracle and agent pipelines."""

import json
import os
import re
from pathlib import Path

from .loops import _LOOP_SKIP_ACTIONS

# ---------------------------------------------------------------------------
# Shared rules and system prompts
# ---------------------------------------------------------------------------

_NO_EFFECT_GROUNDING_RULES = """\
Your job is to identify actions that suffered a **grounding failure** — the action tried \
to interact with a UI element but completely failed to produce any effect. Examples:
- A click that did not register at all (page is truly identical before and after)
- A fill/type action where the text did not appear in the target field
- An action that targeted a non-interactive or wrong element and nothing happened

IMPORTANT: Do NOT flag any of the following as no-effect:
- Actions that trigger a page save, submit, update, or refresh — even if the visible form \
looks similar afterward, these actions commit data to the backend and are meaningful.
- Actions that cause a reload where record IDs, timestamps, or activity logs change — these \
indicate a successful backend operation.
- Actions where a dropdown opened, a menu appeared, navigation occurred, or any UI element \
changed state, even subtly.

Only flag cases where the action genuinely failed to do anything at all."""

_NO_EFFECT_ORACLE_SYSTEM = f"""\
You are an analyst reviewing the effects of UI actions in a web automation trajectory.

For each step you are given a list of candidate actions and a text description of what \
visually changed on the page after each candidate was executed.

{_NO_EFFECT_GROUNDING_RULES}

Respond with valid JSON matching this schema:
{{
  "no_effect_candidates": [
    {{
      "reasoning": "<why this candidate had no effect>",
      "candidate_id": "<the candidate ID string>"
    }}
  ]
}}

If ALL candidates produced meaningful effects, return:
{{"no_effect_candidates": []}}
"""

_NO_EFFECT_AGENT_SYSTEM = f"""\
You are an analyst reviewing a web automation trajectory via screenshots.

You will be shown a sequence of screenshot pairs. For each step, you see:
- The BEFORE screenshot (state when the action was chosen)
- The action that was executed
- The AFTER screenshot (state after the action was executed)

{_NO_EFFECT_GROUNDING_RULES}

Respond with valid JSON matching this schema:
{{
  "no_effect_steps": [
    {{
      "reasoning": "<why this action had no effect>",
      "step_id": "<the step ID string>"
    }}
  ]
}}

If ALL actions produced meaningful effects, return:
{{"no_effect_steps": []}}
"""


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

def _call_openrouter(messages: list[dict], model: str = "openai/gpt-5-mini") -> dict:
    """Call OpenRouter chat completions API and return parsed JSON response."""
    import urllib.request

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    body = json.dumps({
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    content = result["choices"][0]["message"]["content"]
    return json.loads(content)


def _image_to_data_url(image_path: Path) -> str:
    """Convert an image file to a base64 data URL for the OpenRouter vision API."""
    import base64
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Oracle: text-based effects
# ---------------------------------------------------------------------------

def _build_oracle_no_effect_prompt(steps_data: list[dict]) -> str:
    parts = []
    for sd in steps_data:
        parts.append(f"## Step {sd['step']}")
        for c in sd["candidates"]:
            parts.append(f"  Candidate {c['id']}:")
            parts.append(f"    Action: {c['action']}")
            parts.append(f"    Effect: {c['effect']}")
        parts.append("")
    return "\n".join(parts)


def detect_no_effect_oracle(task_name: str, runs_dir: Path) -> list[dict]:
    """Oracle pipeline: use candidate_effects text descriptions."""
    run_task_dir = runs_dir / task_name
    if not run_task_dir.is_dir():
        return []

    steps_data = []
    candidate_registry: dict[str, dict] = {}

    for step_dir in sorted(run_task_dir.iterdir()):
        m = re.match(r"step_(\d+)$", step_dir.name)
        if not m or not step_dir.is_dir():
            continue
        step_idx = int(m.group(1))

        sel_path = step_dir / "selection.json"
        if not sel_path.exists():
            continue
        sel = json.loads(sel_path.read_text())
        selected_idx = sel.get("selected_index")
        selected_from = sel.get("selected_from", "initial")

        effects_path = step_dir / selected_from / "candidate_effects.json"
        candidates_path = step_dir / selected_from / "candidates.json"
        if not effects_path.exists() or not candidates_path.exists():
            continue

        effects = json.loads(effects_path.read_text())
        candidates = json.loads(candidates_path.read_text())

        step_candidates = []
        for j, (cand, effect) in enumerate(zip(candidates, effects)):
            cand_id = f"s{step_idx}_c{j + 1}"
            is_selected = (j + 1) == selected_idx
            action_text = cand.get("action", "")
            step_candidates.append({"id": cand_id, "action": action_text, "effect": effect})
            candidate_registry[cand_id] = {
                "step": step_idx, "action": action_text,
                "effect": effect, "is_selected": is_selected,
            }

        if step_candidates:
            steps_data.append({"step": step_idx, "candidates": step_candidates})

    if not steps_data:
        return []

    BATCH_SIZE = 15
    all_flagged: list[tuple[str, str]] = []

    for batch_start in range(0, len(steps_data), BATCH_SIZE):
        batch = steps_data[batch_start : batch_start + BATCH_SIZE]
        user_prompt = _build_oracle_no_effect_prompt(batch)
        try:
            response = _call_openrouter([
                {"role": "system", "content": _NO_EFFECT_ORACLE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ])
        except Exception as e:
            print(f" [LLM error: {e}]", end="", flush=True)
            continue
        for entry in response.get("no_effect_candidates", []):
            cand_id = entry.get("candidate_id", "")
            reasoning = entry.get("reasoning", "")
            if cand_id in candidate_registry:
                all_flagged.append((cand_id, reasoning))

    no_effect = []
    for cand_id, reasoning in all_flagged:
        info = candidate_registry[cand_id]
        if info["is_selected"]:
            no_effect.append({
                "step": info["step"], "action": info["action"],
                "effect": info["effect"], "reasoning": reasoning,
            })
    return no_effect


# ---------------------------------------------------------------------------
# Agent: vision-based screenshot comparison
# ---------------------------------------------------------------------------

def detect_no_effect_agent(task_dir: Path, actions: list[str]) -> list[dict]:
    """Agent pipeline: compare before/after screenshots via vision LLM.

    For step N, the BEFORE state is screenshot_step_N.png (the observation when
    the agent chose the action) and the AFTER state is screenshot_step_{N+1}.png
    (the observation after the action was executed).
    """
    transitions = []
    for i, action in enumerate(actions):
        if not action:
            continue
        func = action.split("(")[0] if "(" in action else action
        if func in _LOOP_SKIP_ACTIONS or func in ("send_msg_to_user", "report_infeasible"):
            continue
        before = task_dir / f"screenshot_step_{i}.png"
        after = task_dir / f"screenshot_step_{i + 1}.png"
        if before.exists() and after.exists():
            transitions.append((i, action, before, after))

    if not transitions:
        return []

    BATCH_SIZE = 5
    no_effect = []

    for batch_start in range(0, len(transitions), BATCH_SIZE):
        batch = transitions[batch_start : batch_start + BATCH_SIZE]

        content_parts: list[dict] = []
        step_ids = {}

        for step_idx, action, before_path, after_path in batch:
            sid = f"step_{step_idx}"
            step_ids[sid] = {"step": step_idx, "action": action}

            content_parts.append({
                "type": "text",
                "text": f"--- {sid}: Action = {action} ---\nBEFORE:",
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(before_path)},
            })
            content_parts.append({"type": "text", "text": "AFTER:"})
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(after_path)},
            })

        try:
            response = _call_openrouter([
                {"role": "system", "content": _NO_EFFECT_AGENT_SYSTEM},
                {"role": "user", "content": content_parts},
            ])
        except Exception as e:
            print(f" [LLM error: {e}]", end="", flush=True)
            continue

        for entry in response.get("no_effect_steps", []):
            sid = entry.get("step_id", "")
            reasoning = entry.get("reasoning", "")
            if sid in step_ids:
                info = step_ids[sid]
                no_effect.append({
                    "step": info["step"],
                    "action": info["action"],
                    "effect": "(visual comparison)",
                    "reasoning": reasoning,
                })

    return no_effect


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def detect_no_effect_steps(
    task_name: str,
    task_dir: Path,
    pipeline: str,
    actions: list[str],
    runs_dir: Path | None = None,
) -> list[dict]:
    """Dispatch to oracle (text-based) or agent (vision-based) no-effect detection."""
    if pipeline == "oracle" and runs_dir is not None:
        return detect_no_effect_oracle(task_name, runs_dir)
    elif pipeline == "agent":
        return detect_no_effect_agent(task_dir, actions)
    return []
