"""Experiment 2: Interactive AXTree editing.

Replays to a target step, runs a baseline Phase 1, then saves the prompt to a file
for manual editing. After the user edits and presses Enter, sends the edited prompt
to the LLM and compares candidates.

Usage:
    python -m oracle_wm.experiments.axtree_editing \
        --task workarena.servicenow.xxx --seed 50000 --step 0 \
        --run-dir oracle_wm/runs_l3 --headless
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

from agentlab.agents.wm_visual_agent.wm_prompts import CandidateGenerationPrompt
from agentlab.llm.llm_utils import HumanMessage, image_to_jpg_base64_url
from oracle_wm.oracle_pipeline.oracle_loop import (
    _call_llm,
    _dump_prompt,
    _safe_close_env,
    load_committed_from_run,
    replay_committed,
)
from oracle_wm.oracle_pipeline.oracle_prompts import CandidateAwareHistory

from ._common import setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_editable_prompt(text):
    """Parse an editable prompt file back into (system_text, user_text).

    Returns (system_str, user_str). [IMAGE] placeholders are preserved in user_str.
    """
    sys_marker = "=== SYSTEM ==="
    user_marker = "=== USER ==="
    sys_start = text.find(sys_marker)
    user_start = text.find(user_marker)
    if sys_start == -1 or user_start == -1:
        raise ValueError("Could not find === SYSTEM === and === USER === markers")
    system_text = text[sys_start + len(sys_marker):user_start].strip()
    user_text = text[user_start + len(user_marker):].strip()
    return system_text, user_text


def _rebuild_human_message(user_text, original_obs, obs_flags):
    """Rebuild a HumanMessage from edited text + original images.

    Splits user_text on [IMAGE] placeholders and interleaves with original screenshots.
    """
    images = []
    screenshot = original_obs.get("screenshot")
    screenshot_som = original_obs.get("screenshot_som")
    if obs_flags.use_som and screenshot_som is not None:
        images.append(screenshot_som)
    elif obs_flags.use_screenshot and screenshot is not None:
        images.append(screenshot)

    parts = user_text.split("[IMAGE]")
    msg = HumanMessage("")
    img_idx = 0
    for i, part in enumerate(parts):
        if part.strip():
            msg.add_text(part)
        if i < len(parts) - 1 and img_idx < len(images):
            img_url = image_to_jpg_base64_url(images[img_idx])
            msg.add_image(img_url, detail=obs_flags.openai_vision_detail)
            img_idx += 1
    return msg


def main():
    parser = argparse.ArgumentParser(description="Interactive AXTree editing experiment")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--run-dir", default="oracle_wm/runs")
    parser.add_argument("--save-dir", default="oracle_wm/experiments/results/axtree_editing")
    parser.add_argument("--model", default="openai/gpt-5-2025-08-07")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--agent-mode", default="text", choices=["vision", "text"])
    parser.add_argument("--n-candidates", type=int, default=5)
    args = parser.parse_args()

    ctx = setup_experiment(args.task, args.seed, args.model, args.headless, args.agent_mode)
    flags = ctx["flags"]
    oracle_sel_flags = ctx["oracle_sel_flags"]
    action_set = ctx["action_set"]
    obs_preprocessor = ctx["obs_preprocessor"]
    chat_llm = ctx["chat_llm"]
    env_args = ctx["env_args"]
    instance = ctx["instance"]

    run_root = Path(args.run_dir) / f"{args.task.replace('/', '_')}_seed{args.seed}"
    out_dir = Path(args.save_dir) / f"{args.task.replace('/', '_')}_seed{args.seed}_step{args.step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Replay to target step
    if args.step > 0:
        committed, actions_history, thoughts_history, memories_history = load_committed_from_run(run_root, args.step)
    else:
        committed, actions_history, thoughts_history, memories_history = [], [], [], []

    env, obs, current_bid_map, translated_candidate_history = replay_committed(
        env_args, committed, instance, obs_preprocessor
    )
    _safe_close_env(env)
    logger.info(f"Replayed {len(committed)} steps, env closed.")

    gen_flags = oracle_sel_flags if args.agent_mode == "text" else flags
    n_cand = args.n_candidates

    phase1_sys = (
        f"You are an agent trying to solve a web task. Propose your top {n_cand} candidate actions "
        f"for the current state. Balance exploration with exploitation based on the "
        f"semantics/common-sense of the user intent in the task description"
    )

    # Build baseline prompt
    gen_prompt = CandidateGenerationPrompt(
        action_set=action_set,
        obs=obs,
        actions=actions_history,
        thoughts=thoughts_history,
        flags=gen_flags,
        n_candidates=n_cand,
        memories=memories_history,
    )
    if translated_candidate_history:
        gen_prompt.history = CandidateAwareHistory(translated_candidate_history, thoughts_history)

    # Run baseline
    baseline_dir = out_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    _dump_prompt(phase1_sys, gen_prompt.prompt, baseline_dir / "phase1_prompt.txt")
    baseline_response = _call_llm(chat_llm, phase1_sys, gen_prompt.prompt)
    (baseline_dir / "phase1_response.txt").write_text(baseline_response, encoding="utf-8")
    baseline_candidates = gen_prompt.parse_candidates(baseline_response)
    (baseline_dir / "candidates.json").write_text(json.dumps(baseline_candidates, indent=2), encoding="utf-8")
    logger.info(f"Baseline: {len(baseline_candidates)} candidates")

    # Save editable prompt + obs for re-injection
    editable_path = out_dir / "editable_prompt.txt"
    _dump_prompt(phase1_sys, gen_prompt.prompt, editable_path)
    with open(out_dir / "obs.pkl", "wb") as f:
        pickle.dump(obs, f)

    # Wait for user to edit
    print(f"\nPrompt saved to: {editable_path}")
    print(f"Edit the AXTree in the USER section, save, then press Enter to continue...")
    input()

    # Read edited prompt from disk
    edited_text = editable_path.read_text(encoding="utf-8")
    system_text, user_text = _parse_editable_prompt(edited_text)
    human_msg = _rebuild_human_message(user_text, obs, gen_flags.obs)

    # Run edited prompt
    edited_dir = out_dir / "edited"
    edited_dir.mkdir(exist_ok=True)
    _dump_prompt(system_text, human_msg, edited_dir / "phase1_prompt.txt")
    edited_response = _call_llm(chat_llm, system_text, human_msg)
    (edited_dir / "phase1_response.txt").write_text(edited_response, encoding="utf-8")
    edited_candidates = gen_prompt.parse_candidates(edited_response)
    (edited_dir / "candidates.json").write_text(json.dumps(edited_candidates, indent=2), encoding="utf-8")
    logger.info(f"Edited: {len(edited_candidates)} candidates")

    # Save comparison
    comparison = {
        "baseline_candidates": baseline_candidates,
        "edited_candidates": edited_candidates,
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info(f"Comparison saved to {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
