"""Oracle pipeline: agent proposes K candidates, each executed in real env, agent picks one."""

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.wm_visual_agent.agent_configs import (
    DEFAULT_ACTION_FLAGS,
    DEFAULT_OBS_FLAGS,
    DEFAULT_PROMPT_FLAGS,
)
from agentlab.agents.wm_visual_agent.wm_prompts import CandidateGenerationPrompt
from agentlab.experiments.loop import EnvArgs, StepInfo, StepTimestamps
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage

from .oracle_prompts import OracleSelectionPrompt
from .._bid_utils import _ACTION_SET, build_bid_map, get_valid_snow_instance, translate_action

logger = logging.getLogger(__name__)


def _predict_task_username(task_seed: int) -> str:
    """Reproduce the deterministic username that task.setup() would create for the given seed.

    Mirrors the seeding in AbstractServiceNowTask._seed_external_rng() + create_user():
    - np.random.RandomState(seed) is created fresh per task instance, so we replicate it here.
    - Faker.seed(seed) is called again by _seed_external_rng() right before create_user(), so
      calling it here first does not affect what the task will generate.
    """
    import numpy as np
    from faker import Faker

    rng = np.random.RandomState(task_seed)
    Faker.seed(task_seed)
    fake = Faker()
    first_name = fake.first_name()
    last_name = fake.last_name()
    user_idx = str(rng.randint(1000, 9999))
    return f"{first_name}.{last_name}.{user_idx}"


def cleanup_orphaned_users(instance, task_seed: int) -> int:
    """Delete the specific WorkArena task user that would be created for task_seed, if it exists.

    Uses the same deterministic seeding as task.setup() to predict the exact username,
    then deletes only that one record — avoids touching unrelated users.

    Returns 1 if the user was found and deleted, 0 if it didn't exist.
    """
    from browsergym.workarena.api.utils import table_api_call

    username = _predict_task_username(task_seed)
    response = table_api_call(
        instance=instance,
        table="sys_user",
        params={"sysparm_query": f"user_name={username}", "sysparm_fields": "sys_id,user_name"},
        method="GET",
    )
    users = response.get("result", [])
    for user in users:
        table_api_call(instance=instance, table=f"sys_user/{user['sys_id']}", method="DELETE")
        logger.info(f"Deleted orphaned user: {user['user_name']} ({user['sys_id']})")
    if not users:
        logger.info(f"No orphaned user found for seed {task_seed} (username={username!r}).")
    return len(users)


@dataclass
class CommittedAction:
    step: int
    action: str    # action string executed
    bid_map: dict  # bid_map at the decision point (for BID translation on replay)


def _make_env(env_args: EnvArgs, instance):
    return env_args.make_env(
        action_mapping=_ACTION_SET.to_python_code,
        exp_dir=Path("."),
        exp_task_kwargs={"instance": instance},
        use_raw_page_output=False,
    )


def _wait_idle(env):
    try:
        env.unwrapped.page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass


def _dump_prompt(system_text: str, human_msg, path: Path):
    """Serialize a (system, human) message pair to a readable text file.

    Text content is written inline; image content is noted as [IMAGE] placeholders
    so the file is human-readable without embedding base64 blobs.
    """
    lines = ["=== SYSTEM ===", system_text, "", "=== USER ==="]
    content = human_msg.get("content", "")
    if isinstance(content, str):
        lines.append(content)
    else:
        for part in content:
            if part.get("type") == "text":
                lines.append(part["text"])
            elif part.get("type") == "image_url":
                lines.append("[IMAGE]")
    path.write_text("\n".join(lines), encoding="utf-8")


def replay_committed(
    env_args: EnvArgs,
    committed: list[CommittedAction],
    instance,
    max_retries: int = 3,
    retry_delay: float = 30.0,
) -> tuple[object, dict, dict]:
    """Create fresh env, replay all committed actions, return (env, obs, bid_map) at decision point.

    The returned env is open — caller must close it. Retries on transient reset failures.
    Before each retry, cleans up any orphaned user left by a failed partial setup — this handles
    the case where create_user() succeeds but a subsequent role-assignment call fails (502 etc.),
    leaving the deterministic user in ServiceNow without teardown being called.
    """
    obs_preprocessor = dp.make_obs_preprocessor(DEFAULT_OBS_FLAGS)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        env = _make_env(env_args, instance)
        try:
            obs, _ = env.reset(seed=env_args.task_seed)
            break
        except Exception as exc:
            last_exc = exc
            try:
                env.close()
            except Exception:
                pass
            if attempt < max_retries:
                logger.warning(f"env.reset() failed (attempt {attempt}/{max_retries}): {exc}. Cleaning up orphaned user and retrying in {retry_delay}s")
                cleanup_orphaned_users(instance, env_args.task_seed)
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"env.reset() failed after {max_retries} attempts") from last_exc

    _wait_idle(env)
    obs = obs_preprocessor(obs)
    current_bid_map = build_bid_map(obs)

    for ca in committed:
        translated, _ = translate_action(ca.action, ca.bid_map, current_bid_map)
        obs, _, terminated, truncated, _ = env.step(translated)
        if terminated or truncated:
            break
        obs = obs_preprocessor(obs)
        current_bid_map = build_bid_map(obs)

    return env, obs, current_bid_map


def load_committed_from_debug(
    debug_root: Path,
    resume_from: int,
) -> tuple[list[CommittedAction], list[str], list[str]]:
    """Reconstruct committed action history from saved debug files for resume.

    Loads committed_step.json (action=chosen_action, bid_entry from decision_bid_map) so that
    replay_committed can fingerprint-translate BIDs into whatever fresh env is created.
    Also loads selection.json for translated_action (shown in history) and thought.
    """
    committed: list[CommittedAction] = []
    actions_history: list[str] = []
    thoughts_history: list[str] = []
    for i in range(resume_from):
        committed_path = debug_root / f"step_{i}" / "committed_step.json"
        sel_path = debug_root / f"step_{i}" / "selection.json"
        if not committed_path.exists():
            raise FileNotFoundError(f"Cannot resume: missing {committed_path}")
        if not sel_path.exists():
            raise FileNotFoundError(f"Cannot resume: missing {sel_path}")
        step_data = json.loads(committed_path.read_text(encoding="utf-8"))
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        action = step_data["action"]          # chosen_action with original env BIDs
        bid_map = step_data["bid_entry"]      # {orig_bid: entry} from decision_bid_map
        thought = sel.get("thought", "")
        translated = sel.get("translated_action", action)
        committed.append(CommittedAction(step=i, action=action, bid_map=bid_map))
        actions_history.append(translated)    # history shows what was actually executed
        thoughts_history.append(thought)
    logger.info(f"Loaded {len(committed)} committed actions from debug dir for resume")
    return committed, actions_history, thoughts_history


def explore_candidate(
    env_args: EnvArgs,
    committed: list[CommittedAction],
    action: str,
    decision_bid_map: dict,
    instance,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Replay to decision point, execute candidate action, return (screenshot, screenshot_som, bid_note).

    Creates and closes a fresh env sequentially — avoids duplicate data conflicts in ServiceNow
    that would arise from parallel resets of the same deterministic task seed.
    """
    obs_preprocessor = dp.make_obs_preprocessor(DEFAULT_OBS_FLAGS)
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    env, obs, current_bid_map = replay_committed(env_args, committed, instance)
    try:
        translated, bid_note = translate_action(action, decision_bid_map, current_bid_map)
        obs, _, _, _, _ = env.step(translated)
        obs = obs_preprocessor(obs)
        return obs.get("screenshot", blank), obs.get("screenshot_som", blank), bid_note
    finally:
        try:
            env.close()
        except Exception:
            pass


def _call_llm(chat_llm, system_text: str, prompt_msg, n_retry: int = 3) -> str:
    system_prompt = SystemMessage(system_text)
    discussion = Discussion([system_prompt, prompt_msg])
    text = ""
    for attempt in range(1, n_retry + 1):
        response = chat_llm(discussion.messages)
        text = response.get("content", "") if isinstance(response, dict) else getattr(response, "content", str(response))
        if text.strip():
            return text
        if attempt < n_retry:
            logger.warning(f"Empty LLM response, retry {attempt}/{n_retry}")
    return text


def run_oracle_pipeline(
    task_name: str,
    task_seed: int,
    model: str = "openai/gpt-5-2025-08-07",
    n_candidates: int = 5,
    max_steps: int = 30,
    save_dir: str = "oracle_results",
    debug_dir: str = "oracle_wm/debug",
    headless: bool = False,
    cleanup: bool = False,
    resume_from: int = 0,
):
    save_path = Path(save_dir) / f"{task_name.replace('/', '_')}_seed{task_seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    debug_root = Path(debug_dir) / f"{task_name.replace('/', '_')}_seed{task_seed}"
    debug_root.mkdir(parents=True, exist_ok=True)

    flags = DEFAULT_PROMPT_FLAGS
    action_set = DEFAULT_ACTION_FLAGS.action_set.make_action_set()
    chat_llm = CHAT_MODEL_ARGS_DICT[model].make_model()

    env_args = EnvArgs(task_name=task_name, task_seed=task_seed, headless=headless)
    instance = get_valid_snow_instance()

    if cleanup:
        cleanup_orphaned_users(instance, task_seed)

    if resume_from > 0:
        committed, actions_history, thoughts_history = load_committed_from_debug(debug_root, resume_from)
        env, obs, current_bid_map = replay_committed(env_args, committed, instance)
        logger.info(f"Resumed at step {resume_from}: replayed {len(committed)} committed actions")
    else:
        env, obs, current_bid_map = replay_committed(env_args, [], instance)
        committed = []
        actions_history = []
        thoughts_history = []

    step_idx = resume_from
    try:
        while step_idx < max_steps:
            logger.info(f"Step {step_idx}: generating {n_candidates} candidates")
            step_dbg = debug_root / f"step_{step_idx}"
            step_dbg.mkdir(exist_ok=True)
            decision_bid_map = current_bid_map

            # Save current state screenshot + SOM
            sc_current = obs.get("screenshot")
            sc_current_som = obs.get("screenshot_som")
            if sc_current is not None:
                Image.fromarray(sc_current).save(step_dbg / "current.png")
            if sc_current_som is not None:
                Image.fromarray(sc_current_som).save(step_dbg / "current_som.png")

            # Phase 1: generate K candidate actions
            phase1_sys = f"You are an agent trying to solve a web task. Propose your top {n_candidates} candidate actions for the current state."
            gen_prompt = CandidateGenerationPrompt(
                action_set=action_set,
                obs=obs,
                actions=actions_history,
                thoughts=thoughts_history,
                flags=flags,
                n_candidates=n_candidates,
            )
            _dump_prompt(phase1_sys, gen_prompt.prompt, step_dbg / "phase1_prompt.txt")

            phase1_text = _call_llm(chat_llm, phase1_sys, gen_prompt.prompt)
            (step_dbg / "phase1_response.txt").write_text(phase1_text, encoding="utf-8")

            candidates = gen_prompt.parse_candidates(phase1_text)
            if not candidates:
                logger.error("No candidates parsed, stopping.")
                break

            # Write candidates immediately (live debug)
            (step_dbg / "candidates.json").write_text(
                json.dumps(candidates, indent=2), encoding="utf-8"
            )
            logger.info(f"  Saved {len(candidates)} candidates -> {step_dbg / 'candidates.json'}")

            # Close current env before exploration replays
            try:
                env.close()
            except Exception:
                pass
            env = None

            # Phase 2a: explore each candidate sequentially — one fresh env per candidate
            candidate_screenshots: list[np.ndarray] = []
            candidate_screenshots_som: list[np.ndarray] = []
            candidate_bid_notes: list[dict] = []
            for k, cand in enumerate(candidates):
                logger.info(f"  Exploring candidate {k + 1}/{len(candidates)}: {cand['action']!r:.80}")
                sc, sc_som, bid_note = explore_candidate(env_args, committed, cand["action"], decision_bid_map, instance)
                candidate_screenshots.append(sc)
                candidate_screenshots_som.append(sc_som)
                candidate_bid_notes.append({"candidate": k + 1, "action": cand["action"], "bid_note": bid_note})
                # Save future screenshots immediately as they arrive
                Image.fromarray(sc).save(step_dbg / f"step_{step_idx}_future_{k + 1}.png")
                Image.fromarray(sc_som).save(step_dbg / f"step_{step_idx}_future_{k + 1}_som.png")

            (step_dbg / "bid_translations.json").write_text(
                json.dumps(candidate_bid_notes, indent=2), encoding="utf-8"
            )

            # Phase 2b: final replay to decision point, present real SOM screenshots to agent
            env, obs, current_bid_map = replay_committed(env_args, committed, instance)

            # Save the SOM the agent actually sees when making its selection decision
            sc_decision_som = obs.get("screenshot_som")
            if sc_decision_som is not None:
                Image.fromarray(sc_decision_som).save(step_dbg / "decision_point_som.png")

            phase2_sys = "You are an agent trying to solve a web task. Select the best action based on the real environment screenshots."
            oracle_candidates = [(c["action_text"], s) for c, s in zip(candidates, candidate_screenshots_som)]
            sel_prompt = OracleSelectionPrompt(
                obs=obs,
                actions=actions_history,
                thoughts=thoughts_history,
                candidates=oracle_candidates,
                flags=flags,
            )
            _dump_prompt(phase2_sys, sel_prompt.prompt, step_dbg / "phase2_prompt.txt")

            phase2_text = _call_llm(chat_llm, phase2_sys, sel_prompt.prompt)
            (step_dbg / "phase2_response.txt").write_text(phase2_text, encoding="utf-8")

            try:
                selected_idx, reasoning = sel_prompt.parse_answer(phase2_text)
            except (ValueError, ParseError) as e:
                logger.warning(f"  Selection parse failed: {e}, defaulting to candidate 0")
                selected_idx, reasoning = 0, ""

            chosen_action = candidates[selected_idx]["action"]
            logger.info(f"  Selected candidate {selected_idx + 1}: {chosen_action!r:.80}")

            # Translate selected action BIDs for execution.
            # chosen_action has BIDs from decision_bid_map (original live env) — they are consistent.
            # translated_chosen has BIDs from current_bid_map (Phase 2b fresh env).
            obs_preprocessor = dp.make_obs_preprocessor(DEFAULT_OBS_FLAGS)
            translated_chosen, selection_bid_note = translate_action(chosen_action, decision_bid_map, current_bid_map)

            # Extract the BID entry from decision_bid_map for chosen_action's BID.
            # Only this single entry is needed for fingerprint-based translation on future replays.
            _m = re.match(r"\w+\('([^']+)'", chosen_action)
            _bid_entry = {}
            if _m:
                _orig_bid = _m.group(1)
                if _orig_bid in decision_bid_map:
                    _bid_entry = {_orig_bid: decision_bid_map[_orig_bid]}

            # Write selection + resume metadata immediately (live debug)
            (step_dbg / "selection.json").write_text(
                json.dumps({
                    "selected_index": selected_idx + 1,  # 1-indexed for readability
                    "selected_action": chosen_action,
                    "translated_action": translated_chosen,
                    "bid_translation": selection_bid_note,
                    "thought": reasoning,
                }, indent=2),
                encoding="utf-8",
            )
            # committed_step.json stores the data needed to reconstruct CommittedAction on resume:
            # chosen_action (original BIDs) + the bid_entry from decision_bid_map for fingerprinting
            (step_dbg / "committed_step.json").write_text(
                json.dumps({"action": chosen_action, "bid_entry": _bid_entry}, indent=2),
                encoding="utf-8",
            )

            # Execute chosen action on the open env
            t_start = time.time()
            obs, reward, terminated, truncated, env_info = env.step(translated_chosen)
            obs = obs_preprocessor(obs)
            elapsed = time.time() - t_start

            # Build and save StepInfo
            step_info = StepInfo(
                step=step_idx,
                obs=obs.copy(),
                reward=reward,
                raw_reward=env_info.get("RAW_REWARD_GLOBAL", None),
                terminated=terminated,
                truncated=truncated,
                action=translated_chosen,
                agent_info={
                    "think": reasoning,
                    "oracle_candidates": [
                        {"action": c["action"], "action_text": c["action_text"], "rationale": c["rationale"]}
                        for c in candidates
                    ],
                    "oracle_selected_index": selected_idx,
                },
                stats={"step_elapsed": elapsed},
                profiling=StepTimestamps(),
                task_info=env_info.get("task_info", None),
            )

            for k, (sc, sc_som) in enumerate(zip(candidate_screenshots, candidate_screenshots_som)):
                Image.fromarray(sc).save(save_path / f"screenshot_step_{step_idx}_candidate_{k}.png")
                Image.fromarray(sc_som).save(save_path / f"screenshot_som_step_{step_idx}_candidate_{k}.png")

            step_info.save_step_info(save_path, save_screenshot=True, save_som=False)
            logger.info(f"  reward={reward:.3f}  terminated={terminated}  truncated={truncated}")

            # Store chosen_action (original env BIDs) with decision_bid_map (original env bid_map).
            # These are a consistent pair: replay can fingerprint-translate to any future env's BIDs.
            committed.append(CommittedAction(step=step_idx, action=chosen_action, bid_map=decision_bid_map))
            actions_history.append(translated_chosen)
            thoughts_history.append(reasoning)
            current_bid_map = build_bid_map(obs)
            step_idx += 1

            if terminated or truncated:
                logger.info(f"Episode ended at step {step_idx}")
                break

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    logger.info(f"Oracle pipeline complete. {step_idx} steps saved to {save_path}")
    logger.info(f"Debug artifacts saved to {debug_root}")
    return save_path
