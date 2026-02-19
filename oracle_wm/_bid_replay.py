"""Replay and translate-replay modes for BID analysis."""

import json
from pathlib import Path

from agentlab.experiments.loop import get_exp_result

from ._bid_utils import _ACTION_SET, build_bid_map, get_valid_snow_instance, save_som, translate_action


def replay_trajectory(traj_dir: str, output_path: str) -> None:
    """
    Load an existing trajectory, replay its actions on a fresh env (new pool instance),
    and save per-step BID observations.

    The saved JSON contains both the original trajectory's BID data (extracted from its
    stored obs PKL) and the replayed BID data, making self-contained comparison possible.
    """
    exp_result = get_exp_result(traj_dir)
    env_args = exp_result.exp_args.env_args
    task_seed = env_args.task_seed

    original_steps = [
        {
            "action": s.action,
            "url": s.obs.get("url", "") if s.obs else "",
            "bid_map": build_bid_map(s.obs) if s.obs else {},
        }
        for s in exp_result.steps_info
    ]

    instance = get_valid_snow_instance()
    env_args.headless = False
    env = env_args.make_env(
        action_mapping=_ACTION_SET.to_python_code,
        exp_dir=Path("."),
        exp_task_kwargs={"instance": instance},
        use_raw_page_output=False,
    )

    som_dir = Path(output_path).with_suffix("") / "som"
    som_dir.mkdir(parents=True, exist_ok=True)

    total_steps = sum(1 for s in exp_result.steps_info if s.action is not None)
    replayed_steps = []
    try:
        print("reset() ...")
        obs, _ = env.reset(seed=task_seed)
        try:
            env.unwrapped.page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        obs = env.unwrapped._get_obs()
        print(f"  -> url={obs.get('url', '')}  bids={len(obs.get('extra_element_properties', {}))}")
        save_som(obs, som_dir, "step_00_reset")
        replayed_steps.append({
            "action": None,
            "url": obs.get("url", ""),
            "action_error": None,
            "bid_map": build_bid_map(obs),
        })

        step_idx = 0
        for step_info in exp_result.steps_info:
            if step_info.action is None:
                continue
            step_idx += 1
            print(f"step {step_idx}/{total_steps}: {step_info.action!r:.120}")
            obs, _reward, terminated, truncated, _info = env.step(step_info.action)
            err = obs.get("last_action_error") or None
            print(f"  -> url={obs.get('url', '')}  bids={len(obs.get('extra_element_properties', {}))}  err={err!r}")
            save_som(obs, som_dir, f"step_{step_idx:02d}")
            replayed_steps.append({
                "action": step_info.action,
                "url": obs.get("url", ""),
                "action_error": err,
                "bid_map": build_bid_map(obs),
            })
            if terminated or truncated:
                print(f"  -> {'terminated' if terminated else 'truncated'}")
                break
    finally:
        try:
            env.close()
        except AttributeError:
            pass

    out = {
        "task_name": env_args.task_name,
        "task_seed": task_seed,
        "traj_dir": str(traj_dir),
        "original_steps": original_steps,
        "replayed_steps": replayed_steps,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=2))
    print(f"Saved {len(replayed_steps)} replayed steps -> {output_path}")
    print(f"SoM images -> {som_dir}/")


def translate_replay_trajectory(traj_dir: str, output_path: str) -> None:
    """
    Load an existing trajectory, replay it on a fresh env, translating each action's BID
    to the fingerprint-matched BID in the current replay state before executing.

    This allows the replay to succeed even when BIDs drift across pool instances.
    The saved JSON contains original_steps and translated_steps (with both the original
    and translated action per step).
    """
    exp_result = get_exp_result(traj_dir)
    env_args = exp_result.exp_args.env_args
    task_seed = env_args.task_seed

    # orig_step["bid_map"] = obs the agent saw before taking orig_step["action"]
    orig_step_list = [
        {
            "action": s.action,
            "url": s.obs.get("url", "") if s.obs else "",
            "bid_map": build_bid_map(s.obs) if s.obs else {},
        }
        for s in exp_result.steps_info
    ]

    instance = get_valid_snow_instance()
    env_args.headless = False
    env = env_args.make_env(
        action_mapping=_ACTION_SET.to_python_code,
        exp_dir=Path("."),
        exp_task_kwargs={"instance": instance},
        use_raw_page_output=False,
    )

    som_dir = Path(output_path).with_suffix("") / "som"
    som_dir.mkdir(parents=True, exist_ok=True)
    som_dir_orig = Path(output_path).with_suffix("") / "som_original"
    som_dir_orig.mkdir(parents=True, exist_ok=True)

    # Save original SoM images from PKL obs with matching step_XX naming
    _orig_action_idx = 0
    for _s in exp_result.steps_info:
        if _s.obs is None:
            continue
        if _s.action is None:
            save_som(_s.obs, som_dir_orig, "step_00_reset")
        else:
            _orig_action_idx += 1
            save_som(_s.obs, som_dir_orig, f"step_{_orig_action_idx:02d}")

    total_steps = sum(1 for s in orig_step_list if s["action"] is not None)
    translated_steps = []
    try:
        print("reset() ...")
        obs, _ = env.reset(seed=task_seed)
        try:
            env.unwrapped.page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        obs = env.unwrapped._get_obs()
        current_bid_map = build_bid_map(obs)
        print(f"  -> url={obs.get('url', '')}  bids={len(obs.get('extra_element_properties', {}))}")
        save_som(obs, som_dir, "step_00_reset")
        translated_steps.append({
            "action": None,
            "translated_action": None,
            "translation_note": None,
            "url": obs.get("url", ""),
            "action_error": None,
            "bid_map": current_bid_map,
        })

        step_idx = 0
        for orig_step in orig_step_list:
            if orig_step["action"] is None:
                continue
            step_idx += 1
            orig_action = orig_step["action"]
            # orig_step["bid_map"] is the state the original agent was in when it chose this action
            translated_action, note = translate_action(orig_action, orig_step["bid_map"], current_bid_map)
            print(f"step {step_idx}/{total_steps}: {orig_action!r:.100}")
            print(f"  translate: {note}")
            obs, _reward, terminated, truncated, _info = env.step(translated_action)
            err = obs.get("last_action_error") or None
            current_bid_map = build_bid_map(obs)
            print(f"  -> url={obs.get('url', '')}  bids={len(obs.get('extra_element_properties', {}))}  err={err!r}")
            save_som(obs, som_dir, f"step_{step_idx:02d}")
            translated_steps.append({
                "action": orig_action,
                "translated_action": translated_action,
                "translation_note": note,
                "url": obs.get("url", ""),
                "action_error": err,
                "bid_map": current_bid_map,
            })
            if terminated or truncated:
                print(f"  -> {'terminated' if terminated else 'truncated'}")
                break
    finally:
        try:
            env.close()
        except AttributeError:
            pass

    out = {
        "task_name": env_args.task_name,
        "task_seed": task_seed,
        "traj_dir": str(traj_dir),
        "original_steps": orig_step_list,
        "translated_steps": translated_steps,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=2))
    print(f"Saved {len(translated_steps)} translated steps -> {output_path}")
    print(f"SoM images -> {som_dir}/")
