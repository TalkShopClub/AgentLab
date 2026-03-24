"""Output formatting for trajectory analysis reports."""

from .types import TaskAnalysis


def _fmt_action_short(action: str, max_len: int = 60) -> str:
    """Shorten an action string for display."""
    if len(action) <= max_len:
        return action
    return action[: max_len - 3] + "..."


def _fmt_loop(loop: dict) -> str:
    """Format a single loop detection result for display."""
    kind = loop["kind"]
    if kind == "consecutive_repeat":
        action = _fmt_action_short(loop["actions"][0])
        return f"Same action x{loop['repeat_count']} at steps {loop['start_step']}-{loop['start_step'] + loop['repeat_count'] - 1}: {action}"
    elif kind == "consecutive_group":
        actions_str = ", ".join(_fmt_action_short(a, 40) for a in loop["actions"])
        end_step = loop["start_step"] + loop["group_size"] * loop["repeat_count"] - 1
        return (
            f"Group (size {loop['group_size']}) x{loop['repeat_count']} "
            f"at steps {loop['start_step']}-{end_step}: [{actions_str}]"
        )
    elif kind == "non_consecutive_group":
        actions_str = ", ".join(_fmt_action_short(a, 40) for a in loop["actions"])
        occ_str = ", ".join(str(o) for o in loop.get("occurrences", []))
        return (
            f"Non-consecutive group (size {loop['group_size']}, "
            f"{loop['repeat_count']} occurrences at steps [{occ_str}]): [{actions_str}]"
        )
    return str(loop)


def print_report(results: list[TaskAnalysis], results_dir: str):
    """Print the full analysis report to stdout."""
    n = len(results)
    if n == 0:
        print("No tasks found.")
        return

    pipeline = results[0].pipeline
    successes = [r for r in results if r.outcome == "success"]
    n_success = len(successes)
    n_fp = sum(1 for r in results if r.outcome == "false_positive")
    n_max = sum(1 for r in results if r.outcome == "max_steps_exhausted")
    n_loops = sum(1 for r in results if r.has_loops)

    n_consec = sum(1 for r in results if any(l["kind"] == "consecutive_repeat" for l in r.loops))
    n_group = sum(
        1 for r in results
        if any(l["kind"] in ("consecutive_group", "non_consecutive_group") for l in r.loops)
    )

    print(f"\n{'=' * 70}")
    print(f"  Trajectory Analysis")
    print(f"{'=' * 70}")
    print(f"  Dir:      {results_dir}")
    print(f"  Pipeline: {pipeline}  |  Tasks: {n}")
    print()

    print(f"  --- Outcomes ---")
    print(f"    Success:              {n_success:3d} ({n_success / n * 100:5.1f}%)")
    print(f"    False Positive:       {n_fp:3d} ({n_fp / n * 100:5.1f}%)")
    print(f"    Max Steps Exhausted:  {n_max:3d} ({n_max / n * 100:5.1f}%)")
    print()

    if successes:
        lengths = [r.n_steps for r in successes]
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        print(f"  --- Trajectory Length (successful tasks) ---")
        print(f"    Mean: {avg_len:.1f}  |  Min: {min_len}  |  Max: {max_len}")
        print()

    total_looped = sum(r.looped_steps for r in results)
    total_steps = sum(r.n_steps for r in results)
    avg_ratio = total_looped / total_steps if total_steps > 0 else 0.0

    print(f"  --- Loops ---")
    print(f"    Tasks with any loop:    {n_loops:3d} ({n_loops / n * 100:5.1f}%)")
    print(f"    Consecutive repeats:    {n_consec:3d} tasks")
    print(f"    Group repeats:          {n_group:3d} tasks")
    print(f"    Total looped steps:     {total_looped}/{total_steps} ({avg_ratio * 100:5.1f}% of all steps)")
    print()

    tasks_with_no_effect = [r for r in results if r.no_effect_count > 0]
    total_no_effect = sum(r.no_effect_count for r in results)
    if total_no_effect > 0 or any(r.no_effect_steps is not None for r in results):
        print(f"  --- No-Effect Actions (grounding failures) ---")
        print(f"    Tasks with no-effect:   {len(tasks_with_no_effect):3d} ({len(tasks_with_no_effect) / n * 100:5.1f}%)")
        print(f"    Total no-effect steps:  {total_no_effect}/{total_steps} ({total_no_effect / total_steps * 100:5.1f}% of all steps)")
        print()

    print(f"  --- Per Task ---")
    for i, r in enumerate(results, 1):
        short_name = r.task_name
        if len(short_name) > 70:
            short_name = "..." + short_name[-67:]
        print(f"  [{i}] {short_name}")
        print(f"      Outcome: {r.outcome}  |  Steps: {r.n_steps}  |  Reward: {r.reward:.3f}")
        if not r.loops:
            print(f"      Loops: none")
        else:
            print(f"      Looped steps: {r.looped_steps}/{r.n_steps} ({r.looped_step_ratio * 100:.1f}%)")
            for loop in r.loops:
                print(f"      {_fmt_loop(loop)}")
        if r.no_effect_count > 0:
            print(f"      No-effect actions: {r.no_effect_count}")
            for ne in r.no_effect_steps:
                action_short = _fmt_action_short(ne["action"], 50)
                effect_short = _fmt_action_short(ne["effect"], 70)
                reasoning = ne.get("reasoning", "")
                print(f"        step {ne['step']}: {action_short}")
                print(f"          effect: \"{effect_short}\"")
                if reasoning:
                    print(f"          reason: {_fmt_action_short(reasoning, 70)}")
        print()

    print(f"{'=' * 70}")
