"""Oracle pipeline: agent proposes K candidates, each executed in real env, agent picks one."""

import copy
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from bgym import HighLevelActionSetArgs

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent import AGENT_GPT5
from agentlab.agents.wm_visual_agent.agent_configs import (
	DEFAULT_ACTION_FLAGS,
	DEFAULT_OBS_FLAGS,
	DEFAULT_PROMPT_FLAGS,
)
from agentlab.agents.wm_visual_agent.wm_prompts import CandidateGenerationPrompt
from agentlab.experiments.loop import EnvArgs, StepInfo, StepTimestamps
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.llm.llm_utils import Discussion, SystemMessage

from .oracle_prompts import CandidateAwareHistory, CandidateEffectDescriptionPrompt, OracleSelectionPrompt, ResampleRequested
from agentlab.utils.phantom_actions import resolve_phantom_action
from .._bid_utils import _ACTION_SET, _make_env, _wait_idle, build_bid_map, get_valid_snow_instance, translate_action

logger = logging.getLogger(__name__)


def _safe_close_env(env):
    """Close BrowserGym env and ensure the underlying browser process is terminated.

    Grabs a reference to the Playwright Browser before calling env.close(), then
    explicitly closes the browser as a safety net — if env.close() fails (e.g. due to
    a broken transport pipe), this prevents orphaned Chromium processes.
    """
    if env is None:
        return
    browser = None
    try:
        browser = env.unwrapped.page.context.browser
    except Exception:
        pass
    try:
        env.close()
    except Exception:
        pass
    if browser is not None:
        try:
            browser.close()
        except Exception:
            pass


def _predict_task_username(task_seed: int) -> str:
    """Reproduce the deterministic username that task.setup() would create for the given seed.

    Mirrors the seeding in AbstractServiceNowTask._seed_external_rng() + create_user():
    - np.random.RandomState(seed) is created fresh per task instance, so we replicate it here.
    - Faker.seed(seed) is called again by _seed_external_rng() right before create_user(), so
      calling it here first does not affect what the task will generate.
    """
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
    action: str               # action string executed
    bid_map: dict             # single BID entry for chosen action (for translate_action)
    full_decision_bid_map: dict = None  # full map for translating all candidates
    candidates: list = None   # [{action, action_text, rationale}]
    selected_idx: int = -1    # 0-indexed
    selected_from: str = "initial"
    effects: list = None      # per-candidate effect descriptions (parallel to candidates)


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
    obs_preprocessor,
    max_retries: int = 3,
    retry_delay: float = 30.0,
    save_intermediate_soms_to: Path = None,
) -> tuple[object, dict, dict, list[dict]]:
    """Create fresh env, replay all committed actions, return (env, obs, bid_map, candidate_history).

    The returned env is open — caller must close it. Retries on transient reset failures.
    Before each retry, cleans up any orphaned user left by a failed partial setup — this handles
    the case where create_user() succeeds but a subsequent role-assignment call fails (502 etc.),
    leaving the deterministic user in ServiceNow without teardown being called.

    candidate_history contains each committed step's candidates translated into the current
    replay env's BID space, built just before each step's action is executed (the moment when
    current_bid_map equals that step's decision-point BID map).
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        env = _make_env(env_args, instance)
        try:
            obs, _ = env.reset(seed=env_args.task_seed)
            break
        except Exception as exc:
            last_exc = exc
            _safe_close_env(env)
            if attempt < max_retries:
                logger.warning(f"env.reset() failed (attempt {attempt}/{max_retries}): {exc}. Cleaning up orphaned user and retrying in {retry_delay}s")
                cleanup_orphaned_users(instance, env_args.task_seed)
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"env.reset() failed after {max_retries} attempts") from last_exc

    _wait_idle(env)
    obs = obs_preprocessor(obs)
    current_bid_map = build_bid_map(obs)

    translated_candidate_history: list[dict] = []
    for ca in committed:
        # current_bid_map here equals ca's decision-point BID map — translate candidates now
        if save_intermediate_soms_to is not None:
            sc_som = obs.get("screenshot_som")
            if sc_som is not None:
                Image.fromarray(sc_som).save(save_intermediate_soms_to / f"previous_step_{ca.step}_som.png")
        if ca.full_decision_bid_map and ca.candidates:
            translated_cands = []
            for cand in ca.candidates:
                t_action, _ = translate_action(cand["action"], ca.full_decision_bid_map, current_bid_map)
                translated_cands.append({**cand, "action": t_action})
            translated_candidate_history.append({
                "step": ca.step,
                "candidates": translated_cands,
                "selected_idx": ca.selected_idx,
                "effects": ca.effects or [],
            })
        translated, _ = translate_action(ca.action, ca.bid_map, current_bid_map)
        translated = resolve_phantom_action(translated, env)
        obs, _, terminated, truncated, _ = env.step(translated)
        if terminated or truncated:
            break
        _wait_idle(env)
        obs = obs_preprocessor(obs)
        current_bid_map = build_bid_map(obs)

    return env, obs, current_bid_map, translated_candidate_history


def load_committed_from_run(
    run_root: Path,
    resume_from: int,
) -> tuple[list[CommittedAction], list[str], list[str]]:
    """Reconstruct committed action history from saved debug files for resume.

    Loads committed_step.json (action=chosen_action, bid_entry from decision_bid_map) so that
    replay_committed can fingerprint-translate BIDs into whatever fresh env is created.
    Also loads selection.json for translated_action (shown in history) and thought,
    decision_bid_map.json for full candidate translation, and candidates.json for history.
    """
    committed: list[CommittedAction] = []
    actions_history: list[str] = []
    thoughts_history: list[str] = []
    for i in range(resume_from):
        committed_path = run_root / f"step_{i}" / "committed_step.json"
        sel_path = run_root / f"step_{i}" / "selection.json"
        if not committed_path.exists():
            raise FileNotFoundError(f"Cannot resume: missing {committed_path}")
        if not sel_path.exists():
            raise FileNotFoundError(f"Cannot resume: missing {sel_path}")
        step_data = json.loads(committed_path.read_text(encoding="utf-8"))
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        action = step_data["action"]
        bid_map = step_data["bid_entry"]
        thought = sel.get("thought", "")
        translated = sel.get("translated_action", action)
        selected_from = sel.get("selected_from", "initial")
        selected_idx = sel.get("selected_index", 1) - 1

        full_dm = None
        dm_path = run_root / f"step_{i}" / "decision_bid_map.json"
        if dm_path.exists():
            full_dm = json.loads(dm_path.read_text(encoding="utf-8"))

        cands = None
        # New layout: step_N/{attempt}/candidates.json; old layout: step_N/candidates[_resample].json
        cands_path = run_root / f"step_{i}" / selected_from / "candidates.json"
        if not cands_path.exists():
            old_name = "candidates_resample.json" if selected_from == "resample" else "candidates.json"
            cands_path = run_root / f"step_{i}" / old_name
        if cands_path.exists():
            cands = json.loads(cands_path.read_text(encoding="utf-8"))

        effects = None
        effects_path = run_root / f"step_{i}" / selected_from / "candidate_effects.json"
        if not effects_path.exists():
            old_name = "candidate_effects_resample.json" if selected_from == "resample" else "candidate_effects.json"
            effects_path = run_root / f"step_{i}" / old_name
        if effects_path.exists():
            effects = json.loads(effects_path.read_text(encoding="utf-8"))

        committed.append(CommittedAction(
            step=i,
            action=action,
            bid_map=bid_map,
            full_decision_bid_map=full_dm,
            candidates=cands,
            selected_idx=selected_idx,
            selected_from=selected_from,
            effects=effects,
        ))
        actions_history.append(translated)
        thoughts_history.append(thought)
    logger.info(f"Loaded {len(committed)} committed actions from debug dir for resume")
    return committed, actions_history, thoughts_history


def explore_candidate(
    env_args: EnvArgs,
    committed: list[CommittedAction],
    action: str,
    decision_bid_map: dict,
    instance,
    obs_preprocessor,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Replay to decision point, execute candidate action, return (screenshot, screenshot_som, bid_note).

    Creates and closes a fresh env sequentially — avoids duplicate data conflicts in ServiceNow
    that would arise from parallel resets of the same deterministic task seed.
    """
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    env, obs, current_bid_map, _ = replay_committed(env_args, committed, instance, obs_preprocessor)
    try:
        translated, bid_note = translate_action(action, decision_bid_map, current_bid_map)
        translated = resolve_phantom_action(translated, env)
        obs, _, _, _, _ = env.step(translated)
        obs = obs_preprocessor(obs)
        return obs.get("screenshot", blank), obs.get("screenshot_som", blank), bid_note
    finally:
        _safe_close_env(env)


def _explore_on_env(
    env,
    env_args: EnvArgs,
    committed: list[CommittedAction],
    action: str,
    decision_bid_map: dict,
    instance,
    obs_preprocessor,
    max_retries: int = 3,
    retry_delay: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, str, object]:
    """Explore a candidate action by resetting an existing env (reuses the browser process).

    Instead of creating a new browser per candidate, this reuses the given env via reset().
    If reset fails after all retries, the env is replaced with a fresh one.

    Returns (screenshot, screenshot_som, bid_note, env).
    The returned env may differ from the input if recreation was needed.
    """
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            obs, _ = env.reset(seed=env_args.task_seed)
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    f"env.reset() failed (attempt {attempt}/{max_retries}): {exc}. "
                    f"Cleaning up orphaned user and retrying in {retry_delay}s"
                )
                cleanup_orphaned_users(instance, env_args.task_seed)
                time.sleep(retry_delay)
            else:
                # Browser may be corrupted — recreate it
                logger.warning(f"env.reset() failed after {max_retries} attempts, recreating browser")
                _safe_close_env(env)
                env = _make_env(env_args, instance)
                try:
                    obs, _ = env.reset(seed=env_args.task_seed)
                except Exception as final_exc:
                    raise RuntimeError("env.reset() failed even after browser recreation") from final_exc

    _wait_idle(env)
    obs = obs_preprocessor(obs)
    current_bid_map = build_bid_map(obs)

    for ca in committed:
        translated, _ = translate_action(ca.action, ca.bid_map, current_bid_map)
        translated = resolve_phantom_action(translated, env)
        obs, _, terminated, truncated, _ = env.step(translated)
        if terminated or truncated:
            break
        _wait_idle(env)
        obs = obs_preprocessor(obs)
        current_bid_map = build_bid_map(obs)

    translated, bid_note = translate_action(action, decision_bid_map, current_bid_map)
    translated = resolve_phantom_action(translated, env)
    obs, _, _, _, _ = env.step(translated)
    obs = obs_preprocessor(obs)

    return obs.get("screenshot", blank), obs.get("screenshot_som", blank), bid_note, env


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
    run_dir: str = "oracle_wm/runs",
    headless: bool = False,
    cleanup: bool = False,
    resume_from: int = 0,
    agent_mode: str = "vision",  # "vision" = SOM screenshot, "text" = AXTree only
    sel_effects: bool = True,
    sel_images: bool = True,
):
    save_path = Path(save_dir) / f"{task_name.replace('/', '_')}_seed{task_seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    run_root = Path(run_dir) / f"{task_name.replace('/', '_')}_seed{task_seed}"
    run_root.mkdir(parents=True, exist_ok=True)

    if agent_mode == "text":
        flags = copy.deepcopy(AGENT_GPT5.flags)
        flags.action.long_description = True
        action_set = HighLevelActionSetArgs(subsets=["bid"]).make_action_set()
    else:
        flags = DEFAULT_PROMPT_FLAGS
        action_set = DEFAULT_ACTION_FLAGS.action_set.make_action_set()

    # Oracle selection always shows plain screenshot of current state (no SOM overlay).
    # This is independent of agent_mode: text mode needs use_screenshot enabled;
    # vision mode needs use_som disabled so only the plain image is shown.
    # For text mode, use_ax_tree is already True on oracle_sel_flags (inherited from AGENT_GPT5).
    oracle_sel_flags = copy.deepcopy(flags)
    oracle_sel_flags.obs.use_screenshot = True
    oracle_sel_flags.obs.use_som = False

    # Preprocessor used at every env interaction. Always extracts AXTree + SOM + screenshot so
    # that text-mode agents have axtree_txt available for Phase 1, and SOM images are always
    # stored for debug HTML regardless of agent_mode.
    preproc_obs_flags = copy.deepcopy(DEFAULT_OBS_FLAGS)
    preproc_obs_flags.use_ax_tree = True
    obs_preprocessor = dp.make_obs_preprocessor(preproc_obs_flags)

    chat_llm = CHAT_MODEL_ARGS_DICT[model].make_model()

    env_args = EnvArgs(task_name=task_name, task_seed=task_seed, headless=headless)
    instance = get_valid_snow_instance()

    if cleanup:
        cleanup_orphaned_users(instance, task_seed)

    if resume_from > 0:
        committed, actions_history, thoughts_history = load_committed_from_run(run_root, resume_from)
        env, obs, current_bid_map, translated_candidate_history = replay_committed(env_args, committed, instance, obs_preprocessor)
        logger.info(f"Resumed at step {resume_from}: replayed {len(committed)} committed actions")
    else:
        env, obs, current_bid_map, translated_candidate_history = replay_committed(env_args, [], instance, obs_preprocessor)
        committed = []
        actions_history = []
        thoughts_history = []

    goal_text = obs.get("goal", "")
    if goal_text:
        (run_root / "goal.txt").write_text(goal_text, encoding="utf-8")

    step_idx = resume_from
    reward, terminated, truncated = 0.0, False, False
    try:
        while step_idx < max_steps:
            logger.info(f"Step {step_idx}: generating {n_candidates} candidates")
            step_dir = run_root / f"step_{step_idx}"
            step_dir.mkdir(exist_ok=True)
            decision_bid_map = current_bid_map

            # Close current env before exploration replays
            _safe_close_env(env)
            env = None

            # ── generate → explore → select (with one resample attempt) ──
            resample_reasoning = ""
            selected_from = "initial"
            for attempt_tag in ("initial", "resample"):
                is_resample = attempt_tag == "resample"
                attempt_dir = step_dir / attempt_tag
                attempt_dir.mkdir(exist_ok=True)

                # Snapshot bid map matching the obs Phase 1 will see.
                # For initial: current_bid_map == decision_bid_map (live env).
                # For resample: current_bid_map is the Phase 2b initial map — the env whose
                # AXTree produced the BIDs the LLM writes in candidates. Using decision_bid_map
                # (live env) here would cause translate_action to look up BIDs in the wrong map.
                generation_bid_map = current_bid_map

                # Save current state (obs used by Phase 1 generation for this attempt).
                # For initial: the live env state. For resample: the Phase 2b decision point from initial.
                sc_cur = obs.get("screenshot")
                sc_cur_som = obs.get("screenshot_som")
                if sc_cur is not None:
                    Image.fromarray(sc_cur).save(attempt_dir / "current.png")
                if sc_cur_som is not None:
                    Image.fromarray(sc_cur_som).save(attempt_dir / "current_som.png")

                # Phase 1: generate K candidate actions
                phase1_sys = f"You are an agent trying to solve a web task. Propose your top {n_candidates} candidate actions for the current state."
                if is_resample:
                    phase1_sys += (
                        f"\n\nIMPORTANT — your previous set of candidates was rejected for the following reason:\n"
                        f"> {resample_reasoning}\n\n"
                        f"Generate a completely new set of candidates that addresses the issues above. "
                        f"Focus on:\n"
                        f"- Targeting different elements (different BIDs) than the rejected set.\n"
                        f"- Re-examining BID grounding — verify each BID actually corresponds to "
                        f"the UI element you intend to interact with.\n"
                        f"- Trying fundamentally different strategies, not minor variations."
                    )

                # Text mode: oracle_sel_flags adds plain screenshot to AXTree context.
                # Vision mode: original flags keeps SOM overlay for BID identification.
                gen_flags = oracle_sel_flags if agent_mode == "text" else flags
                gen_prompt = CandidateGenerationPrompt(
                    action_set=action_set,
                    obs=obs,
                    actions=actions_history,
                    thoughts=thoughts_history,
                    flags=gen_flags,
                    n_candidates=n_candidates,
                )
                if translated_candidate_history:
                    gen_prompt.history = CandidateAwareHistory(translated_candidate_history, thoughts_history)
                _dump_prompt(phase1_sys, gen_prompt.prompt, attempt_dir / "phase1_prompt.txt")

                phase1_text = _call_llm(chat_llm, phase1_sys, gen_prompt.prompt)
                (attempt_dir / "phase1_response.txt").write_text(phase1_text, encoding="utf-8")

                candidates = gen_prompt.parse_candidates(phase1_text)
                if not candidates:
                    raise RuntimeError("No candidates parsed from LLM response, cannot continue.")

                (attempt_dir / "candidates.json").write_text(
                    json.dumps(candidates, indent=2), encoding="utf-8"
                )
                logger.info(f"  Saved {len(candidates)} candidates -> {attempt_tag}/candidates.json")

                # Phase 2a: explore each candidate (reusing single browser)
                candidate_screenshots: list[np.ndarray] = []
                candidate_screenshots_som: list[np.ndarray] = []
                candidate_bid_notes: list[dict] = []
                exploration_env = _make_env(env_args, instance)
                try:
                    for k, cand in enumerate(candidates):
                        logger.info(f"  Exploring candidate {k + 1}/{len(candidates)}: {cand['action']!r:.80}")
                        sc, sc_som, bid_note, exploration_env = _explore_on_env(
                            exploration_env, env_args, committed, cand["action"],
                            generation_bid_map, instance, obs_preprocessor,
                        )
                        candidate_screenshots.append(sc)
                        candidate_screenshots_som.append(sc_som)
                        candidate_bid_notes.append({"candidate": k + 1, "action": cand["action"], "bid_note": bid_note})
                        Image.fromarray(sc).save(attempt_dir / f"future_{k + 1}.png")
                        Image.fromarray(sc_som).save(attempt_dir / f"future_{k + 1}_som.png")
                finally:
                    _safe_close_env(exploration_env)
                    exploration_env = None

                (attempt_dir / "bid_translations.json").write_text(
                    json.dumps(candidate_bid_notes, indent=2), encoding="utf-8"
                )

                # Phase 2a.5: describe visual effect of each candidate
                cand_images = candidate_screenshots if agent_mode == "text" else candidate_screenshots_som
                effect_sys = "You are comparing browser states. Describe what changed after each action."
                effect_prompt = CandidateEffectDescriptionPrompt(obs, candidates, cand_images, oracle_sel_flags)
                effect_text = _call_llm(chat_llm, effect_sys, effect_prompt.prompt)
                candidate_effects = effect_prompt.parse_effects(effect_text)
                (attempt_dir / "candidate_effects.json").write_text(
                    json.dumps(candidate_effects, indent=2), encoding="utf-8"
                )

                # Phase 2b: replay to decision point for selection
                env, obs, current_bid_map, translated_candidate_history = replay_committed(env_args, committed, instance, obs_preprocessor, save_intermediate_soms_to=step_dir)

                sc_decision_som = obs.get("screenshot_som")
                if sc_decision_som is not None:
                    Image.fromarray(sc_decision_som).save(attempt_dir / "decision_point_som.png")

                phase2_sys = "You are an agent trying to solve a web task. Select the best action based on the real environment screenshots."
                oracle_candidates = [
                    (
                        c["action_text"],
                        translate_action(c["action"], generation_bid_map, current_bid_map)[0],
                        s,
                    )
                    for c, s in zip(candidates, cand_images)
                ]
                allow_resample = not is_resample  # allow resample only on first attempt
                sel_prompt = OracleSelectionPrompt(
                    obs=obs,
                    actions=actions_history,
                    thoughts=thoughts_history,
                    candidates=oracle_candidates,
                    flags=oracle_sel_flags,
                    allow_resample=allow_resample,
                    effects=candidate_effects,
                    include_effects=sel_effects,
                    include_images=sel_images,
                )
                _dump_prompt(phase2_sys, sel_prompt.prompt, attempt_dir / "phase2_prompt.txt")

                phase2_text = _call_llm(chat_llm, phase2_sys, sel_prompt.prompt)
                (attempt_dir / "phase2_response.txt").write_text(phase2_text, encoding="utf-8")

                try:
                    selected_idx, reasoning = sel_prompt.parse_answer(phase2_text)
                    selected_from = attempt_tag
                    break  # selection made, exit the generate-select loop
                except ResampleRequested as e:
                    resample_reasoning = e.reasoning
                    logger.info(f"  Resample requested: {resample_reasoning[:120]}")
                    # Close the env from phase 2b before re-generating
                    _safe_close_env(env)
                    env = None
                    continue  # go to resample attempt

            chosen_action = candidates[selected_idx]["action"]
            logger.info(f"  Selected candidate {selected_idx + 1}: {chosen_action!r:.80}")

            # Translate selected action BIDs for execution.
            # chosen_action has BIDs from decision_bid_map (original live env) — they are consistent.
            # translated_chosen has BIDs from current_bid_map (Phase 2b fresh env).
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
            (step_dir / "selection.json").write_text(
                json.dumps({
                    "selected_index": selected_idx + 1,  # 1-indexed for readability
                    "selected_from": selected_from,       # "initial" or "resample"
                    "selected_action": chosen_action,
                    "translated_action": translated_chosen,
                    "bid_translation": selection_bid_note,
                    "thought": reasoning,
                }, indent=2),
                encoding="utf-8",
            )
            # committed_step.json stores the data needed to reconstruct CommittedAction on resume:
            # chosen_action (original BIDs) + the bid_entry from decision_bid_map for fingerprinting
            (step_dir / "committed_step.json").write_text(
                json.dumps({"action": chosen_action, "bid_entry": _bid_entry}, indent=2),
                encoding="utf-8",
            )
            (step_dir / "decision_bid_map.json").write_text(
                json.dumps(decision_bid_map, indent=2), encoding="utf-8"
            )

            # Execute chosen action on the open env (wait for page to settle first)
            _wait_idle(env)
            t_start = time.time()
            translated_chosen = resolve_phantom_action(translated_chosen, env)
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

            step_info.save_step_info(save_path, save_screenshot=True, save_som=False)
            logger.info(f"  reward={reward:.3f}  terminated={terminated}  truncated={truncated}")

            # Store chosen_action with bid_entry (single BID for compat) and full decision_bid_map
            # (for translating all candidates on replay).
            committed.append(CommittedAction(
                step=step_idx,
                action=chosen_action,
                bid_map=_bid_entry,
                full_decision_bid_map=decision_bid_map,
                candidates=candidates,
                selected_idx=selected_idx,
                selected_from=selected_from,
                effects=candidate_effects,
            ))
            actions_history.append(translated_chosen)
            thoughts_history.append(reasoning)
            current_bid_map = build_bid_map(obs)
            step_idx += 1

            if terminated or truncated:
                logger.info(f"Episode ended at step {step_idx}")
                break

    finally:
        _safe_close_env(env)

    summary = {
        "task_name": task_name,
        "task_seed": task_seed,
        "steps": step_idx,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }
    with open(save_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Oracle pipeline complete. {step_idx} steps, reward={reward:.3f}. Run artifacts: {run_root}")
    return save_path, reward
