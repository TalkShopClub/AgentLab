#!/usr/bin/env python3
"""
Replay world model predictions on existing trajectory steps.

This script:
1. Loads a JSON file with list of {traj_path, step_id} entries
2. For each entry, loads that step's state from the trajectory
3. Reconstructs the context (goal, history) up to that step
4. Uses the same WM agent prompts to generate candidates and predictions
5. Saves results in a format compatible with create_html_from_results.py

Usage:
    python replay_wm_predictions.py --input trajectories.json --output-dir wm_replay_results/
"""

import argparse
import base64
import gzip
import io
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from browsergym.experiments import get_exp_result
from tqdm import tqdm

from agentlab.agents.wm_visual_agent.agent_configs import (
    DEFAULT_ACTION_FLAGS,
    DEFAULT_OBS_FLAGS,
    DEFAULT_PROMPT_FLAGS,
)
from agentlab.agents.wm_visual_agent.wm_prompts import (
    CandidateGenerationPrompt,
    InformedSelectionPrompt,
)
from agentlab.agents.world_model_client import WorldModelClient
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.llm.llm_utils import Discussion, SystemMessage

# Import from parent directory script
sys.path.insert(0, str(Path(__file__).parent.parent))
from create_videos_from_results import extract_goal_from_axtree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trajectory_step(
    traj_path: str, step_id: int, level: str = "l3"
) -> Dict[str, Any]:
    """Load trajectory data up to and including the specified step.

    Returns:
        Dict with keys:
            - obs: observation at step_id
            - goal: task goal
            - actions: list of actions taken before step_id
            - thoughts: list of thoughts before step_id
            - original_action: the actual action taken at step_id in the trajectory
            - original_thought: the actual thought at step_id in the trajectory
            - traj_path: path to trajectory
            - step_id: the step ID
            - task_name: name of the task
    """
    exp_result = get_exp_result(traj_path)

    if step_id >= len(exp_result.steps_info):
        raise ValueError(
            f"step_id {step_id} out of range (trajectory has {len(exp_result.steps_info)} steps)"
        )

    step_info = exp_result.steps_info[step_id]
    obs = step_info.obs

    # Extract goal using existing function
    goal = "Goal not available"
    if level == "l2":
        goal_file = Path(traj_path) / "goal_object.pkl.gz"
        if goal_file.exists():
            try:
                with gzip.open(goal_file, "rb") as f:
                    goal_obj = pickle.load(f)
                if isinstance(goal_obj, tuple) and len(goal_obj) > 0:
                    if isinstance(goal_obj[0], dict) and "text" in goal_obj[0]:
                        goal = goal_obj[0]["text"]
            except Exception as e:
                logger.warning(f"Could not load goal from goal_object.pkl.gz: {e}")
    else:
        if "axtree_txt" in obs:
            goal = extract_goal_from_axtree(obs["axtree_txt"])

    # Extract history (actions and thoughts) up to but not including step_id
    actions = []
    thoughts = []
    for i in range(step_id):
        si = exp_result.steps_info[i]
        actions.append(si.action if hasattr(si, "action") else None)
        if hasattr(si, "agent_info") and hasattr(si.agent_info, "think"):
            thoughts.append(si.agent_info.think)
        else:
            thoughts.append(None)

    # Get the ORIGINAL action and thought at step_id (what the agent actually did)
    original_action = step_info.action if hasattr(step_info, "action") else None
    original_thought = None
    if hasattr(step_info, "agent_info") and hasattr(step_info.agent_info, "think"):
        original_thought = step_info.agent_info.think

    # Get task name
    task_name = "Unknown"
    if hasattr(exp_result, "exp_args"):
        exp_args = exp_result.exp_args
        if hasattr(exp_args, "env_args"):
            task_name = getattr(exp_args.env_args, "task_name", "Unknown")

    return {
        "obs": obs,
        "goal": goal,
        "actions": actions,
        "thoughts": thoughts,
        "original_action": original_action,
        "original_thought": original_thought,
        "traj_path": traj_path,
        "step_id": step_id,
        "task_name": task_name,
    }


def generate_candidates(
    obs: dict,
    actions: list,
    thoughts: list,
    chat_llm,
    flags,
    action_set,
    model_name: str,
    output_dir: Path,
    traj_name: str,
    step_id: int,
) -> List[Dict]:
    """Generate candidate actions using Phase 1 prompt."""
    gen_prompt = CandidateGenerationPrompt(
        action_set=action_set,
        obs=obs,
        actions=actions,
        thoughts=thoughts,
        flags=flags,
        model_name=model_name,
    )

    system_prompt = SystemMessage(
        "You are an agent trying to solve a web task. "
        "Propose your top 5 candidate actions for the current state."
    )

    chat_messages = Discussion([system_prompt, gen_prompt.prompt])

    response = chat_llm(chat_messages.messages)
    text = response.content if hasattr(response, "content") else str(response)
    if hasattr(response, "choices"):
        text = response.choices[0].message.content

    candidates = gen_prompt.parse_candidates(text)

    # Save candidates as JSON
    candidates_file = output_dir / f"{traj_name}_step_{step_id}_phase1_candidates.json"
    with open(candidates_file, "w") as f:
        json.dump({"raw_response": text, "parsed_candidates": candidates}, f, indent=2)
    logger.info(f"  Saved Phase 1 candidates to {candidates_file.name}")

    return candidates


def run_phase2_selection(
    obs: dict,
    candidates: List[Dict],
    predictions: List[Dict],
    chat_llm,
    flags,
    action_set,
    model_name: str,
    screenshot_som: np.ndarray,
    output_dir: Path,
    traj_name: str,
    step_id: int,
) -> Dict[str, Any]:
    """Phase 2: Agent selects best action after seeing WM predictions."""
    # Create a modified obs with SOM screenshot for Phase 2
    obs_som = obs.copy()
    obs_som["screenshot"] = screenshot_som
    obs_som["screenshot_som"] = screenshot_som

    sel_prompt = InformedSelectionPrompt(
        action_set=action_set,
        obs=obs_som,
        candidates=candidates,
        predictions=predictions,
        flags=flags,
        mode="text",  # We always use text mode in replay
        model_name=model_name,
    )

    system_prompt = SystemMessage(
        "You are an agent trying to solve a web task. "
        "A world model has predicted the next state for each candidate action that was picked by you earlier. "
        "Select the best action based on the predictions, such that it will help you go in the correct direction for solving the task."
    )

    chat_messages = Discussion([system_prompt, sel_prompt.prompt])

    response = chat_llm(chat_messages.messages)
    text = response.content if hasattr(response, "content") else str(response)
    if hasattr(response, "choices"):
        text = response.choices[0].message.content

    # Parse the response
    try:
        ans_dict = sel_prompt.parse_answer(text)
        result = {
            "selected_action": ans_dict.get("action"),
            "selection_thought": ans_dict.get("think"),
        }
    except Exception as e:
        logger.warning(f"  Failed to parse Phase 2 selection: {e}")
        result = {
            "selected_action": None,
            "selection_thought": None,
        }

    # Save selection as JSON
    selection_file = output_dir / f"{traj_name}_step_{step_id}_phase2_selection.json"
    with open(selection_file, "w") as f:
        json.dump({"raw_response": text, "parsed_selection": result}, f, indent=2)
    logger.info(f"  Saved Phase 2 selection to {selection_file.name}")

    return result


def process_single_step(
    traj_path: str,
    step_id: int,
    wm_client: WorldModelClient,
    chat_llm,
    flags,
    action_set,
    model_name: str,
    output_dir: Path,
    level: str = "l3",
) -> Dict[str, Any]:
    """Process a single step: load context, generate candidates, get predictions."""
    traj_name = Path(traj_path).name
    logger.info(f"Processing {traj_name} step {step_id}")

    # Load trajectory data
    step_data = load_trajectory_step(traj_path, step_id, level)

    obs = step_data["obs"]
    actions = step_data["actions"]
    thoughts = step_data["thoughts"]

    # Check if candidates already exist
    candidates_file = output_dir / f"{traj_name}_step_{step_id}_phase1_candidates.json"
    candidates = None

    if candidates_file.exists():
        logger.info(f"  Found existing candidates file: {candidates_file.name}")
        with open(candidates_file, "r") as f:
            candidates_data = json.load(f)
            candidates = candidates_data.get("parsed_candidates", [])
        if candidates:
            logger.info(f"  Loaded {len(candidates)} candidates from file")
        else:
            logger.warning("  Existing file has no candidates, check...")
            exit(0)

    # Generate candidates with retry logic if not loaded from file
    if not candidates:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            logger.info(f"  Generating candidates (attempt {attempt}/{max_retries})...")
            candidates = generate_candidates(
                obs, actions, thoughts, chat_llm, flags, action_set, model_name,
                output_dir, traj_name, step_id
            )

            if candidates:
                logger.info(f"  Generated {len(candidates)} candidates")
                break
            else:
                if attempt < max_retries:
                    logger.warning(f"  No candidates generated on attempt {attempt}, retrying...")
                else:
                    logger.error(f"  Failed to generate candidates after {max_retries} attempts!")
                    return None

    # Get REGULAR screenshot for Emu3.5 (better text reading)
    screenshot_regular = obs.get("screenshot")
    if screenshot_regular is None:
        logger.warning("  No regular screenshot available!")
        return None

    # Get SOM screenshot for Phase 2 agent selection
    screenshot_som = obs.get("screenshot_som")
    if screenshot_som is None:
        logger.warning("  No SOM screenshot available!")
        screenshot_som = screenshot_regular  # Fallback to regular

    # Get world model predictions using REGULAR screenshot
    logger.info("  Getting world model predictions (using regular screenshot)...")
    action_texts = [c["action_text"] for c in candidates]
    predictions = wm_client.predict_batch(screenshot_regular, action_texts)

    logger.info("  Running Phase 2: Agent selection with WM predictions...")
    # Phase 2: Let agent pick best action after seeing predictions (using SOM screenshot)
    agent_selection = run_phase2_selection(
        obs, candidates, predictions, chat_llm, flags, action_set, model_name, screenshot_som,
        output_dir, traj_name, step_id
    )

    logger.info("  Done!")

    # Convert SOM screenshot to base64 for storage (for HTML visualization)
    img_som = Image.fromarray(screenshot_som)
    if img_som.mode in ("RGBA", "LA"):
        img_som = img_som.convert("RGB")
    buf = io.BytesIO()
    img_som.save(buf, format="PNG")
    screenshot_som_b64 = base64.b64encode(buf.getvalue()).decode()

    # Also save regular screenshot for reference
    img_reg = Image.fromarray(screenshot_regular)
    if img_reg.mode in ("RGBA", "LA"):
        img_reg = img_reg.convert("RGB")
    buf = io.BytesIO()
    img_reg.save(buf, format="PNG")
    screenshot_regular_b64 = base64.b64encode(buf.getvalue()).decode()

    # Convert prediction images to base64
    predictions_serializable = []
    for pred in predictions:
        pred_copy = pred.copy()
        if pred_copy.get("image") is not None:
            img_array = pred_copy["image"]
            img = Image.fromarray(img_array.astype("uint8"))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            pred_copy["image"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        predictions_serializable.append(pred_copy)

    return {
        "traj_path": step_data["traj_path"],
        "step_id": step_data["step_id"],
        "task_name": step_data["task_name"],
        "goal": step_data["goal"],
        "original_action": step_data["original_action"],
        "original_thought": step_data["original_thought"],
        "candidates": candidates,
        "predictions": predictions_serializable,
        "agent_selection": agent_selection,
        "screenshot_som_b64": screenshot_som_b64,
        "screenshot_regular_b64": screenshot_regular_b64,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Replay world model predictions on trajectory steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example input JSON format:
[
    {
        "traj_path": "/path/to/trajectory_folder_1",
        "step_id": 5
    },
    {
        "traj_path": "/path/to/trajectory_folder_2",
        "step_id": 10
    }
]
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="JSON file with list of {traj_path, step_id} entries",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save prediction results",
    )
    parser.add_argument(
        "--wm-server-url",
        type=str,
        default="https://z66y0a4p8qruii-8000.proxy.runpod.net/",
        help="World model server URL",
    )
    parser.add_argument(
        "--wm-mode",
        type=str,
        default="text",
        choices=["text", "image"],
        help="World model prediction mode",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="l2",
        choices=["l2", "l3"],
        help="Task level (affects goal extraction)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5-2025-08-07",
        help="LLM model for candidate generation",
    )

    args = parser.parse_args()

    # Load input JSON
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r") as f:
        entries = json.load(f)

    logger.info(f"Loaded {len(entries)} entries from {input_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize world model client
    wm_client = WorldModelClient(
        server_url=args.wm_server_url, mode=args.wm_mode, timeout=600
    )

    # Check WM server health
    if not wm_client.health_check():
        logger.error(
            f"World model server not responding at {args.wm_server_url}. Please start the server."
        )
        sys.exit(1)

    logger.info(f"Connected to world model server: {args.wm_server_url}")

    # Initialize LLM
    chat_model_args = CHAT_MODEL_ARGS_DICT[args.model]
    chat_llm = chat_model_args.make_model()

    # Initialize action set and flags
    flags = DEFAULT_PROMPT_FLAGS
    action_set = DEFAULT_ACTION_FLAGS.action_set.make_action_set()

    # Process each entry
    results = []
    for entry in tqdm(entries, desc="Processing steps"):
        traj_path = entry["traj_path"]
        step_id = entry["step_id"]

        try:
            result = process_single_step(
                traj_path,
                step_id,
                wm_client,
                chat_llm,
                flags,
                action_set,
                args.model,
                output_dir,
                args.level,
            )

            if result:
                results.append(result)

                # Save individual result
                traj_name = Path(traj_path).name
                result_file = output_dir / f"{traj_name}_step_{step_id}.json"
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)

        except Exception as e:
            logger.error(f"Error processing {traj_path} step {step_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save combined results
    output_file = output_dir / "all_predictions.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Processed {len(results)}/{len(entries)} steps successfully")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
