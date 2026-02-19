"""Shared utilities for BID analysis: fingerprinting, SoM saving, action translation."""

import re
from collections import defaultdict
from pathlib import Path

from bgym import HighLevelActionSetArgs
from PIL import Image
from browsergym.utils.obs import overlay_som

import browsergym.workarena  # noqa: F401  # pyright: ignore[reportUnusedImport]
from browsergym.workarena.instance import SNowInstance

SNAPSHOTS_DIR = Path(__file__).parent / "bid_snapshots"

# Same action set as WMVisualAgent (agent_configs.py:22-26)
_ACTION_SET = HighLevelActionSetArgs(subsets=["coord", "bid"]).make_action_set()


def build_bid_map(obs: dict) -> dict:
    """Merge extra_element_properties with AXTree nodes into BID -> element record.

    Fingerprinting strategy (applied in order until unique):
      1. role::name[::level][::placeholder]
      2. If ambiguous, enrich with full named ancestor path + nearest preceding named sibling
      3. If still ambiguous, assign y-rank (sort by DOM snapshot bbox y-coordinate)
    """
    nodeid_to_node: dict[str, dict] = {}
    nodeid_to_parentid: dict[str, str] = {}
    bid_to_ax: dict[str, dict] = {}

    for node in obs.get("axtree_object", {}).get("nodes", []):
        nid = node.get("nodeId", "")
        if nid:
            nodeid_to_node[nid] = node
            pid = node.get("parentId")
            if pid:
                nodeid_to_parentid[nid] = pid
        bid = node.get("browsergym_id")
        if not bid:
            continue
        props_dict: dict[str, str] = {}
        for p in node.get("properties", []):
            v = p.get("value", {}).get("value")
            if v not in (None, "", False):
                props_dict[p["name"]] = str(v)
        bid_to_ax[bid] = {
            "role": node.get("role", {}).get("value", ""),
            "name": node.get("name", {}).get("value", ""),
            "description": node.get("description", {}).get("value", ""),
            "value": node.get("value", {}).get("value", ""),
            "props": props_dict,
            "_nodeid": nid,
        }

    def _make_fp_base(ax: dict) -> str:
        fp = f"{ax.get('role', '')}::{ax.get('name', '')}"
        p = ax.get("props", {})
        if "level" in p:
            fp += f"::level={p['level']}"
        if "placeholder" in p:
            fp += f"::placeholder={p['placeholder']}"
        return fp

    def _full_ancestor_path(nodeid: str, own_name: str) -> str:
        """Walk the full AXTree parent chain, joining all distinct named ancestors into a path."""
        parts: list[str] = []
        visited: set[str] = set()
        current = nodeid_to_parentid.get(nodeid)
        while current and current not in visited:
            visited.add(current)
            anc_name = nodeid_to_node.get(current, {}).get("name", {}).get("value", "")
            if anc_name and anc_name != own_name:
                parts.append(anc_name)
            current = nodeid_to_parentid.get(current)
        return "/".join(reversed(parts)) if parts else ""

    def _preceding_named_sibling(nodeid: str, own_name: str) -> str:
        """Return the nearest preceding sibling with a distinct non-empty name.
        Uses childIds from the parent node directly (Chrome-provided order).
        """
        parent_id = nodeid_to_parentid.get(nodeid)
        if not parent_id:
            return ""
        children = nodeid_to_node.get(parent_id, {}).get("childIds", [])
        try:
            idx = children.index(nodeid)
        except ValueError:
            return ""
        for i in range(idx - 1, -1, -1):
            sib_name = nodeid_to_node.get(children[i], {}).get("name", {}).get("value", "")
            if sib_name and sib_name != own_name:
                return sib_name
        return ""

    extra = obs.get("extra_element_properties", {})

    # Pass 1: count base fingerprint occurrences
    fp_counts: dict[str, int] = {}
    for bid in extra:
        k = _make_fp_base(bid_to_ax.get(bid, {}))
        fp_counts[k] = fp_counts.get(k, 0) + 1

    # Pass 2: enrich ambiguous base fps with full ancestor path + preceding sibling
    bid_to_fp: dict[str, str] = {}
    enriched_counts: dict[str, int] = {}
    for bid in extra:
        ax = bid_to_ax.get(bid, {})
        fp_base = _make_fp_base(ax)
        if fp_counts[fp_base] > 1:
            nid = ax.get("_nodeid", "")
            own_name = ax.get("name", "")
            anc = _full_ancestor_path(nid, own_name)
            sib = _preceding_named_sibling(nid, own_name)
            fp = fp_base
            if anc:
                fp += f"::via={anc}"
            if sib:
                fp += f"::after={sib}"
        else:
            fp = fp_base
        bid_to_fp[bid] = fp
        enriched_counts[fp] = enriched_counts.get(fp, 0) + 1

    # Pass 3: y-rank for still-ambiguous fps (sort by DOM snapshot bbox y, stable per seed)
    fp_to_bids: dict[str, list[str]] = defaultdict(list)
    for bid, fp in bid_to_fp.items():
        if enriched_counts[fp] > 1:
            fp_to_bids[fp].append(bid)

    bid_to_yrank: dict[str, int] = {}
    for fp, bids in fp_to_bids.items():
        def _y(b: str) -> float:
            bbox = extra.get(b, {}).get("bbox")
            return bbox[1] if bbox else float("inf")
        for rank, b in enumerate(sorted(bids, key=_y), 1):
            bid_to_yrank[b] = rank

    bid_map: dict[str, dict] = {}
    for bid, eprops in extra.items():
        ax = bid_to_ax.get(bid, {})
        fp = bid_to_fp.get(bid, _make_fp_base(ax))
        if enriched_counts[fp] > 1:
            fp = f"{fp}::yrank={bid_to_yrank.get(bid, 0)}"
        bid_map[bid] = {
            "role": ax.get("role", ""),
            "name": ax.get("name", ""),
            "description": ax.get("description", ""),
            "bbox": eprops.get("bbox"),
            "visibility": eprops.get("visibility"),
            "clickable": eprops.get("clickable"),
            "fingerprint": fp,
        }
    return bid_map


def save_som(obs: dict, som_dir: Path, name: str) -> None:
    """Save a Set-of-Marks annotated screenshot. BBoxes are scaled 0.5x (2560->1280)."""
    screenshot = obs.get("screenshot")
    extra = obs.get("extra_element_properties", {})
    if screenshot is None or not extra:
        return
    scaled = {}
    for bid, props in extra.items():
        p = props.copy()
        if props.get("bbox") is not None:
            x, y, w, h = props["bbox"]
            p["bbox"] = [x * 0.5, y * 0.5, w * 0.5, h * 0.5]
        scaled[bid] = p
    som = overlay_som(screenshot, extra_properties=scaled)
    Image.fromarray(som).save(som_dir / f"{name}.png")


def translate_action(action: str, orig_bid_map: dict, replay_bid_map: dict) -> tuple[str, str]:
    """
    Translate an action's BID from the original trajectory to the corresponding BID in
    the replay environment, using fingerprint matching.

    Returns (translated_action, note).
    """
    m = re.match(r"\w+\('([^']+)'", action)
    if not m:
        return action, "no-bid-in-action"
    orig_bid = m.group(1)
    entry = orig_bid_map.get(orig_bid)
    if not entry:
        return action, f"orig_bid={orig_bid!r} not in orig_bid_map"
    fp = entry["fingerprint"]
    fp_to_bid = {v["fingerprint"]: bid for bid, v in replay_bid_map.items()}
    replay_bid = fp_to_bid.get(fp)
    if not replay_bid:
        return action, f"fp={fp!r} not found in replay, keeping original BID"
    if orig_bid == replay_bid:
        return action, f"stable fp={fp!r}"
    return action.replace(f"'{orig_bid}'", f"'{replay_bid}'", 1), \
           f"translated {orig_bid!r} -> {replay_bid!r} via fp={fp!r}"


def get_valid_snow_instance() -> SNowInstance:
    """Pick a real service-now.com instance from the pool (retry up to 10x)."""
    for attempt in range(10):
        instance = SNowInstance()
        if "service-now.com" in instance.snow_url:
            print(f"Using instance: {instance.snow_url}")
            return instance
        print(f"Skipping non-ServiceNow URL: {instance.snow_url} (attempt {attempt + 1})")
    raise RuntimeError("Could not get a valid ServiceNow instance after 10 attempts")
