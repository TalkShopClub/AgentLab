"""Loop detection: consecutive repeats, consecutive groups, non-consecutive groups."""

# Actions excluded from loop detection (navigation noise, not meaningful loops)
_LOOP_SKIP_ACTIONS = {"tab_focus", "noop"}


def detect_consecutive_repeats(actions: list[str]) -> list[dict]:
    """Detect runs of the same action repeated consecutively (A->A->A)."""
    results = []
    i = 0
    n = len(actions)
    while i < n:
        j = i + 1
        while j < n and actions[j] == actions[i]:
            j += 1
        count = j - i
        if count >= 2:
            results.append({
                "kind": "consecutive_repeat",
                "actions": [actions[i]],
                "start_step": i,
                "repeat_count": count,
                "group_size": 1,
            })
        i = j
    return results


def detect_consecutive_group_repeats(actions: list[str]) -> list[dict]:
    """Detect consecutive group repeats (ABCD->ABCD). Largest groups found first."""
    n = len(actions)
    results = []
    covered = set()

    for g in range(n // 2, 1, -1):
        i = 0
        while i <= n - 2 * g:
            if i in covered:
                i += 1
                continue
            group = tuple(actions[i : i + g])
            repeat_count = 1
            pos = i + g
            while pos + g <= n and tuple(actions[pos : pos + g]) == group:
                repeat_count += 1
                pos += g
            if repeat_count >= 2:
                span = set(range(i, i + g * repeat_count))
                if not span.issubset(covered):
                    results.append({
                        "kind": "consecutive_group",
                        "actions": list(group),
                        "start_step": i,
                        "repeat_count": repeat_count,
                        "group_size": g,
                    })
                    covered.update(span)
            i += 1
    return results


def detect_non_consecutive_group_repeats(actions: list[str]) -> list[dict]:
    """Detect non-consecutive group repeats (ABCD->EF->ABCD).

    Scans largest groups first and suppresses sub-groups whose step ranges
    are already covered by a larger detected group.
    """
    n = len(actions)
    results = []
    covered_ranges: list[set[int]] = []

    def _is_subsumed(occurrences: list[int], g: int) -> bool:
        for occ_start in occurrences:
            occ_range = set(range(occ_start, occ_start + g))
            if not any(occ_range.issubset(cr) for cr in covered_ranges):
                return False
        return True

    for g in range(n // 2, 1, -1):
        seen: dict[tuple, list[int]] = {}
        for i in range(n - g + 1):
            key = tuple(actions[i : i + g])
            seen.setdefault(key, []).append(i)

        for group_tuple, positions in seen.items():
            if len(positions) < 2:
                continue
            non_overlapping = [positions[0]]
            for p in positions[1:]:
                if p >= non_overlapping[-1] + g:
                    non_overlapping.append(p)
            if len(non_overlapping) < 2:
                continue

            all_consecutive = all(
                non_overlapping[j + 1] == non_overlapping[j] + g
                for j in range(len(non_overlapping) - 1)
            )
            if all_consecutive:
                continue

            if _is_subsumed(non_overlapping, g):
                continue

            results.append({
                "kind": "non_consecutive_group",
                "actions": list(group_tuple),
                "start_step": non_overlapping[0],
                "repeat_count": len(non_overlapping),
                "group_size": g,
                "occurrences": non_overlapping,
            })
            for occ_start in non_overlapping:
                covered_ranges.append(set(range(occ_start, occ_start + g)))

    return results


def _filter_for_loops(actions: list[str]) -> tuple[list[str], list[int]]:
    """Filter out noise actions (tab_focus, noop) for loop detection.

    Returns (filtered_actions, original_indices) so loop results can be
    mapped back to the original step numbering.
    """
    filtered = []
    indices = []
    for i, action in enumerate(actions):
        func = action.split("(")[0] if "(" in action else action
        if func not in _LOOP_SKIP_ACTIONS:
            filtered.append(action)
            indices.append(i)
    return filtered, indices


def _remap_loop(loop: dict, index_map: list[int]) -> dict:
    """Remap step indices in a loop result back to original step numbering."""
    remapped = loop.copy()
    remapped["start_step"] = index_map[loop["start_step"]]
    if "occurrences" in remapped:
        remapped["occurrences"] = [index_map[o] for o in remapped["occurrences"]]
    return remapped


def detect_all_loops(actions: list[str]) -> list[dict]:
    """Run all loop detectors on filtered actions, remap to original step indices."""
    filtered, index_map = _filter_for_loops(actions)

    raw_loops = []
    raw_loops.extend(detect_consecutive_repeats(filtered))
    raw_loops.extend(detect_consecutive_group_repeats(filtered))
    raw_loops.extend(detect_non_consecutive_group_repeats(filtered))

    return [_remap_loop(l, index_map) for l in raw_loops]


def compute_looped_steps(loops: list[dict], n_steps: int) -> tuple[int, float]:
    """Count steps consumed by non-overlapping loops, largest-first greedy.

    For each loop, the "extra" steps beyond a single occurrence are the looped
    steps (i.e. the repetitions). For a consecutive repeat of count C, the extra
    steps are C-1. For a group of size G repeated R times, extra = G*(R-1).
    For non-consecutive groups, each occurrence beyond the first contributes G steps.

    We greedily assign steps to the largest loops first (by total extra steps)
    and skip steps already claimed.

    Returns (looped_steps, ratio).
    """
    if not loops or n_steps == 0:
        return 0, 0.0

    entries: list[tuple[int, set[int]]] = []
    for loop in loops:
        kind = loop["kind"]
        g = loop["group_size"]
        start = loop["start_step"]

        if kind == "consecutive_repeat":
            count = loop["repeat_count"]
            extra = set(range(start + 1, start + count))
        elif kind == "consecutive_group":
            count = loop["repeat_count"]
            extra = set(range(start + g, start + g * count))
        elif kind == "non_consecutive_group":
            occs = loop.get("occurrences", [])
            extra = set()
            for occ in occs[1:]:
                extra.update(range(occ, occ + g))
        else:
            continue

        entries.append((len(extra), extra))

    entries.sort(key=lambda e: e[0], reverse=True)

    claimed: set[int] = set()
    for _, extra_steps in entries:
        claimed.update(extra_steps - claimed)

    looped = len(claimed)
    return looped, looped / n_steps if n_steps > 0 else 0.0
