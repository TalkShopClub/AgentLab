"""Experiment 1: Forced resample loop.

Replays to a cherry-picked step, then repeatedly asks the LLM to propose candidates
while rejecting all previous rounds as "dead ends". Each round's candidates are explored
in the real environment to capture future screenshots for subjective evaluation.

Usage:
    python -m oracle_wm.experiments.forced_resample \
        --task workarena.servicenow.xxx --seed 50000 --step 2 \
        --n-rounds 5 --run-dir oracle_wm/runs_l3 --headless
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from agentlab.agents.wm_visual_agent.wm_prompts import CandidateGenerationPrompt
from oracle_wm._bid_utils import _make_env
from oracle_wm.oracle_pipeline.oracle_loop import (
    _call_llm,
    _dump_prompt,
    _explore_on_env,
    _safe_close_env,
    load_committed_from_run,
    replay_committed,
)
from oracle_wm.oracle_pipeline.oracle_prompts import CandidateAwareHistory

from ._common import setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Forced resample loop experiment")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True, help="Step to resample at")
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--run-dir", default="oracle_wm/runs")
    parser.add_argument("--save-dir", default="oracle_wm/experiments/results/forced_resample")
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

    # Load committed history and replay to target step
    committed, actions_history, thoughts_history, memories_history = load_committed_from_run(run_root, args.step)
    env, obs, current_bid_map, translated_candidate_history = replay_committed(
        env_args, committed, instance, obs_preprocessor
    )
    generation_bid_map = current_bid_map
    _safe_close_env(env)
    logger.info(f"Replayed {len(committed)} steps, env closed.")

    gen_flags = oracle_sel_flags if args.agent_mode == "text" else flags
    n_cand = args.n_candidates
    all_rounds = []
    accumulated_history = list(translated_candidate_history)
    accumulated_thoughts = list(thoughts_history)

    for round_i in range(args.n_rounds):
        round_dir = out_dir / f"round_{round_i}"
        round_dir.mkdir(exist_ok=True)

        # Build system prompt
        phase1_sys = (
            f"You are an agent trying to solve a web task. Propose your top {n_cand} candidate actions "
            f"for the current state. Balance exploration with exploitation based on the "
            f"semantics/common-sense of the user intent in the task description"
        )
        if round_i > 0:
            phase1_sys += (
                "\n\nIMPORTANT — your previous set of candidates was rejected for the following reason:\n"
                "> All proposed candidates are dead ends that do not make meaningful progress toward the goal.\n\n"
                "Generate a completely new set of candidates that addresses the issues above. "
                "Focus on:\n"
                "- Targeting different elements (different BIDs) than the rejected set.\n"
                "- Re-examining BID grounding — verify each BID actually corresponds to "
                "the UI element you intend to interact with.\n"
                "- Trying fundamentally different strategies, not minor variations."
            )

        # Build prompt
        gen_prompt = CandidateGenerationPrompt(
            action_set=action_set,
            obs=obs,
            actions=actions_history,
            thoughts=thoughts_history,
            flags=gen_flags,
            n_candidates=n_cand,
            memories=memories_history,
        )
        if accumulated_history:
            gen_prompt.history = CandidateAwareHistory(accumulated_history, accumulated_thoughts)

        # Save prompt, call LLM, save response
        _dump_prompt(phase1_sys, gen_prompt.prompt, round_dir / "phase1_prompt.txt")
        response_text = _call_llm(chat_llm, phase1_sys, gen_prompt.prompt)
        (round_dir / "phase1_response.txt").write_text(response_text, encoding="utf-8")

        candidates = gen_prompt.parse_candidates(response_text)
        (round_dir / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")

        logger.info(f"Round {round_i}: {len(candidates)} candidates")
        for j, c in enumerate(candidates):
            logger.info(f"  C{j+1}: {c['action']}")

        # Explore each candidate in real env → save future screenshots
        exploration_env = _make_env(env_args, instance)
        try:
            for k, cand in enumerate(candidates):
                logger.info(f"  Exploring C{k+1}: {cand['action']!r:.80}")
                sc, sc_som, bid_note, exploration_env = _explore_on_env(
                    exploration_env, env_args, committed, cand["action"],
                    generation_bid_map, instance, obs_preprocessor,
                )
                Image.fromarray(sc).save(round_dir / f"future_{k+1}.png")
                Image.fromarray(sc_som).save(round_dir / f"future_{k+1}_som.png")
        finally:
            _safe_close_env(exploration_env)

        all_rounds.append({"round": round_i, "candidates": candidates})

        # Accumulate history for next round (mark as rejected)
        accumulated_history.append({
            "step": args.step,
            "candidates": candidates,
            "selected_idx": -1,
            "effects": ["REJECTED"] * len(candidates),
        })
        accumulated_thoughts.append("REJECTED — forced resample")

    (out_dir / "all_rounds.json").write_text(json.dumps(all_rounds, indent=2), encoding="utf-8")

    all_actions = [c["action"] for r in all_rounds for c in r["candidates"]]
    unique_actions = set(all_actions)
    logger.info(f"Done. {args.n_rounds} rounds, {len(all_actions)} total, {len(unique_actions)} unique actions")


if __name__ == "__main__":
    main()
