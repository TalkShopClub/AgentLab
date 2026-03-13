"""Experiment 3: Curated action selection with full oracle exploration.

Replays to a target step, dumps AXTree + BID map + screenshots for the user to examine,
waits for manually curated candidates, then explores them in real env, describes effects,
and presents the full oracle selection prompt (with future images + effects + resample option).

Usage:
    python -m oracle_wm.experiments.curated_selection \
        --task workarena.servicenow.xxx --seed 50000 --step 2 \
        --run-dir oracle_wm/runs_l3 --headless
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from oracle_wm._bid_utils import _make_env, translate_action
from oracle_wm.oracle_pipeline.oracle_loop import (
    _call_llm,
    _dump_prompt,
    _explore_on_env,
    _safe_close_env,
    load_committed_from_run,
    replay_committed,
)
from oracle_wm.oracle_pipeline.oracle_prompts import (
    CandidateEffectDescriptionPrompt,
    OracleSelectionPrompt,
    ResampleRequested,
)

from ._common import setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEMPLATE_CANDIDATES = [
    {"action": "click('BID')", "action_text": "Description of what this does"},
]


def main():
    parser = argparse.ArgumentParser(description="Curated action selection experiment")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--run-dir", default="oracle_wm/runs")
    parser.add_argument("--save-dir", default="oracle_wm/experiments/results/curated_selection")
    parser.add_argument("--model", default="openai/gpt-5-2025-08-07")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--agent-mode", default="text", choices=["vision", "text"])
    args = parser.parse_args()

    ctx = setup_experiment(args.task, args.seed, args.model, args.headless, args.agent_mode)
    flags = ctx["flags"]
    oracle_sel_flags = ctx["oracle_sel_flags"]
    obs_preprocessor = ctx["obs_preprocessor"]
    chat_llm = ctx["chat_llm"]
    env_args = ctx["env_args"]
    instance = ctx["instance"]

    run_root = Path(args.run_dir) / f"{args.task.replace('/', '_')}_seed{args.seed}"
    out_dir = Path(args.save_dir) / f"{args.task.replace('/', '_')}_seed{args.seed}_step{args.step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load committed history and replay to target step
    committed, actions_history, thoughts_history, memories_history = load_committed_from_run(run_root, args.step)
    env, obs, current_bid_map, _ = replay_committed(
        env_args, committed, instance, obs_preprocessor
    )
    generation_bid_map = current_bid_map

    # Dump artifacts for user inspection
    (out_dir / "axtree.txt").write_text(obs.get("axtree_txt", ""), encoding="utf-8")
    (out_dir / "decision_bid_map.json").write_text(json.dumps(generation_bid_map, indent=2), encoding="utf-8")
    (out_dir / "goal.txt").write_text(obs.get("goal", ""), encoding="utf-8")
    sc = obs.get("screenshot")
    sc_som = obs.get("screenshot_som")
    if sc is not None:
        Image.fromarray(sc).save(out_dir / "current.png")
    if sc_som is not None:
        Image.fromarray(sc_som).save(out_dir / "current_som.png")

    # Write template candidates file
    candidates_path = out_dir / "curated_candidates.json"
    if not candidates_path.exists():
        candidates_path.write_text(json.dumps(TEMPLATE_CANDIDATES, indent=2), encoding="utf-8")

    _safe_close_env(env)
    logger.info(f"Replayed {len(committed)} steps. Artifacts saved to {out_dir}")

    print(f"\nArtifacts saved to: {out_dir}")
    print(f"  axtree.txt             — AXTree at decision point")
    print(f"  decision_bid_map.json  — BID map with fingerprints")
    print(f"  current.png / current_som.png — screenshots")
    print(f"  curated_candidates.json — EDIT THIS with your candidates")
    print(f"\nWrite your curated candidates, save, then press Enter...")
    input()

    # Read curated candidates
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(candidates)} curated candidates")

    # Phase 2a: explore each candidate in real env
    explore_dir = out_dir / "explore"
    explore_dir.mkdir(exist_ok=True)
    candidate_screenshots = []
    candidate_screenshots_som = []
    bid_notes = []

    exploration_env = _make_env(env_args, instance)
    try:
        for k, cand in enumerate(candidates):
            logger.info(f"  Exploring C{k+1}: {cand['action']!r:.80}")
            sc, sc_som, bid_note, exploration_env = _explore_on_env(
                exploration_env, env_args, committed, cand["action"],
                generation_bid_map, instance, obs_preprocessor,
            )
            candidate_screenshots.append(sc)
            candidate_screenshots_som.append(sc_som)
            bid_notes.append({"candidate": k + 1, "action": cand["action"], "bid_note": bid_note})
            Image.fromarray(sc).save(explore_dir / f"future_{k+1}.png")
            Image.fromarray(sc_som).save(explore_dir / f"future_{k+1}_som.png")
    finally:
        _safe_close_env(exploration_env)

    (explore_dir / "bid_translations.json").write_text(json.dumps(bid_notes, indent=2), encoding="utf-8")

    # Phase 2a.5: describe effects
    cand_images = candidate_screenshots if args.agent_mode == "text" else candidate_screenshots_som
    effect_sys = "You are comparing browser states. Describe what changed after each action."
    effect_prompt = CandidateEffectDescriptionPrompt(obs, candidates, cand_images, oracle_sel_flags)
    _dump_prompt(effect_sys, effect_prompt.prompt, explore_dir / "effects_prompt.txt")
    effect_text = _call_llm(chat_llm, effect_sys, effect_prompt.prompt)
    (explore_dir / "effects_response.txt").write_text(effect_text, encoding="utf-8")
    candidate_effects = effect_prompt.parse_effects(effect_text)
    (explore_dir / "candidate_effects.json").write_text(json.dumps(candidate_effects, indent=2), encoding="utf-8")

    # Phase 2b: replay again to decision point for selection
    env, obs, current_bid_map, _ = replay_committed(env_args, committed, instance, obs_preprocessor)

    sc_decision_som = obs.get("screenshot_som")
    if sc_decision_som is not None:
        Image.fromarray(sc_decision_som).save(out_dir / "decision_point_som.png")

    phase2_sys = (
        "You are an agent trying to solve a web task. "
        "Select the best action based on the real environment screenshots."
    )
    oracle_candidates = [
        (
            c.get("action_text", ""),
            translate_action(c["action"], generation_bid_map, current_bid_map)[0],
            s,
        )
        for c, s in zip(candidates, cand_images)
    ]
    sel_prompt = OracleSelectionPrompt(
        obs=obs,
        actions=actions_history,
        thoughts=thoughts_history,
        candidates=oracle_candidates,
        flags=oracle_sel_flags,
        allow_resample=True,
        effects=candidate_effects,
        include_effects=True,
        include_images=True,
    )
    _dump_prompt(phase2_sys, sel_prompt.prompt, out_dir / "phase2_prompt.txt")
    phase2_text = _call_llm(chat_llm, phase2_sys, sel_prompt.prompt)
    (out_dir / "phase2_response.txt").write_text(phase2_text, encoding="utf-8")

    try:
        selected_idx, reasoning = sel_prompt.parse_answer(phase2_text)
        chosen = candidates[selected_idx]
        selection = {
            "selected_index": selected_idx + 1,
            "selected_action": chosen["action"],
            "selected_action_text": chosen.get("action_text", ""),
            "thought": reasoning,
            "resample": False,
        }
        logger.info(f"Selected C{selected_idx + 1}: {chosen['action']}")
    except ResampleRequested as e:
        selection = {
            "resample": True,
            "resample_reasoning": e.reasoning,
        }
        logger.info(f"Agent requested resample: {e.reasoning[:120]}")
    except Exception as e:
        selection = {"error": str(e)}
        logger.error(f"Failed to parse selection: {e}")

    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    _safe_close_env(env)


if __name__ == "__main__":
    main()
