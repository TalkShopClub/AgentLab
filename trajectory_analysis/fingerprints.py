"""Action normalization via fingerprint-based BID replacement."""

import re

# Actions that don't use BIDs (first quoted arg is not a BID)
_NO_BID_ACTIONS = {"send_msg_to_user", "report_infeasible", "scroll", "noop"}


def normalize_action(action: str, bid_map: dict) -> str:
    """Replace BID in an action string with its fingerprint for stable comparison.

    E.g. click('123') -> click(fp='button::Submit')
    """
    if not action:
        return action

    func_name = action.split("(")[0] if "(" in action else action
    if func_name in _NO_BID_ACTIONS:
        return action

    m = re.match(r"(\w+)\('([^']+)'", action)
    if not m:
        return action

    bid = m.group(2)
    entry = bid_map.get(bid)
    if not entry or "fingerprint" not in entry:
        return action

    fp = entry["fingerprint"]
    return action.replace(f"'{bid}'", f"fp='{fp}'", 1)


def extract_step_data(steps: list, skip_fingerprint: bool) -> tuple[
    list[str],   # normalized actions
    list[bool],  # terminated flags
    list[bool],  # truncated flags
]:
    """Extract normalized actions and flags from steps, computing fingerprints if needed."""
    from oracle_wm._bid_utils import build_bid_map

    actions = []
    terminated_flags = []
    truncated_flags = []

    for step in steps:
        raw_action = step.action
        if raw_action is None:
            continue

        if skip_fingerprint:
            actions.append(raw_action)
        else:
            obs = step.obs if isinstance(step.obs, dict) else {}
            if obs.get("axtree_object") and obs.get("extra_element_properties"):
                bid_map = build_bid_map(obs)
                actions.append(normalize_action(raw_action, bid_map))
            else:
                actions.append(raw_action)

        terminated_flags.append(bool(step.terminated))
        truncated_flags.append(bool(step.truncated))

    return actions, terminated_flags, truncated_flags
