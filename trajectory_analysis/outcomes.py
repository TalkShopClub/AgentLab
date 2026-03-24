"""Task outcome classification."""


def classify_outcome(
    actions: list[str],
    terminated_flags: list[bool],
    truncated_flags: list[bool],
    reward: float,
) -> str:
    if reward > 0:
        return "success"

    for i, action in enumerate(actions):
        if action and "send_msg_to_user(" in action:
            # FP: episode didn't end at this step
            if not terminated_flags[i] and not truncated_flags[i]:
                return "false_positive"
            # FP: there are more steps after this one
            if i < len(actions) - 1:
                return "false_positive"
            break

    return "max_steps_exhausted"
