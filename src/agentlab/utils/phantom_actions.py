"""Phantom DOM element resolution for BrowserGym environments.

ServiceNow (and other SPAs) render accessible elements as tiny/invisible anchor nodes
(e.g. 1×1 px) with the actual visual widget on a parent container. Playwright synthesises
mouse events at the geometric center of the BID element, so click/hover on a phantom lands
on a near-invisible pixel and has no effect.

resolve_phantom_action() is a drop-in pre-processor for env.step(action): it rewrites
click/dblclick/right_click/hover actions that target phantom BIDs to target the resolved
visual container instead. It is safe to call for any env type — non-browser envs and
resolution failures both return the original action unchanged.
"""

import re

_PHANTOM_MIN_AREA = 25       # anything <= this is a phantom anchor (e.g. 1×1, 2×2)
_PHANTOM_MAX_AREA = 400 * 60  # anything >= this spans multiple widgets (full row/toolbar)
_PHANTOM_MAX_DEPTH = 5        # max DOM levels to walk upward

# Mouse-position actions where a phantom BID causes a mis-click; keyboard actions (press)
# are excluded because keyboard events bubble through the DOM regardless of element size.
_PHANTOM_VERB_MAP = {
    "click":       "mouse_click",
    "dblclick":    "mouse_dblclick",
    "right_click": "mouse_right_click",
    "hover":       "mouse_hover",
}


def _resolve_clickable_bbox(elem) -> tuple[dict | None, int, str | None]:
    """Return (bbox, depth, parent_bid) for the first DOM ancestor with a widget-sized
    bbox that also contains the original element's center point.

    Uses Playwright's ElementHandle.bounding_box() for every ancestor rather than JS
    getBoundingClientRect(), so coordinates are always in main-page space regardless of
    iframes. The containment check uses those same main-page coords.

    Returns (bbox, 0, None)      if the element itself already has a valid area.
    Returns (bbox, depth>=1, bid) if a containing DOM ancestor was used.
    Returns (None, -1, None)     if nothing suitable found within _PHANTOM_MAX_DEPTH levels.
    """
    raw_bbox = elem.bounding_box()
    if raw_bbox is not None:
        area = raw_bbox["width"] * raw_bbox["height"]
        if _PHANTOM_MIN_AREA < area < _PHANTOM_MAX_AREA:
            return raw_bbox, 0, None

    if raw_bbox is None:
        return None, -1, None

    orig_cx = raw_bbox["x"] + raw_bbox["width"] / 2
    orig_cy = raw_bbox["y"] + raw_bbox["height"] / 2

    cur = elem
    for depth in range(1, _PHANTOM_MAX_DEPTH + 1):
        try:
            parent = cur.evaluate_handle("el => el.parentElement")
        except Exception:
            break
        if parent is None:
            break

        parent_bbox = parent.bounding_box()
        if parent_bbox is not None:
            area = parent_bbox["width"] * parent_bbox["height"]
            if _PHANTOM_MIN_AREA < area < _PHANTOM_MAX_AREA:
                # Containment check in main-page coords — >= to allow center ON edge
                if (orig_cx >= parent_bbox["x"] and
                        orig_cx <= parent_bbox["x"] + parent_bbox["width"] and
                        orig_cy >= parent_bbox["y"] and
                        orig_cy <= parent_bbox["y"] + parent_bbox["height"]):
                    try:
                        parent_bid = parent.evaluate(
                            "el => el.getAttribute('browsergym_id')"
                            " || el.getAttribute('data-id')"
                            " || el.getAttribute('data-testid') || null"
                        )
                    except Exception:
                        parent_bid = None
                    return parent_bbox, depth, parent_bid

        cur = parent

    return None, -1, None


def resolve_phantom_action(action: str, env) -> str:
    """Rewrite a mouse action that targets a phantom BID to target its visual container.

    A phantom element has a bounding box area <= _PHANTOM_MIN_AREA (e.g. 1×1 accessibility
    anchors in ServiceNow). Playwright synthesises mouse events at the element's geometric
    center, so clicking/hovering a phantom lands on a near-invisible pixel.

    Resolution strategy (in order):
      1. If the resolved ancestor has a browsergym_id, substitute BID in-place:
           click('phantom') -> click('parent_bid')
      2. Otherwise fall back to a coordinate action (coord subset must be in action set):
           click('phantom') -> mouse_click(cx, cy)

    Only verbs in _PHANTOM_VERB_MAP are processed; press/fill/select_option are left
    unchanged because keyboard events bubble through the DOM regardless of element size.
    Safe to call for any env type — returns original action unchanged if the env has no
    browser page, the element is not a phantom, or resolution fails.
    """
    m = re.match(r"^(\w+)\('([^']+)'", action)
    if not m or m.group(1) not in _PHANTOM_VERB_MAP:
        return action
    verb, bid = m.group(1), m.group(2)

    try:
        page = env.unwrapped.page
    except AttributeError:
        return action

    try:
        from browsergym.core.action.utils import get_elem_by_bid
        elem = get_elem_by_bid(page, bid)
    except Exception:
        return action

    raw_bbox = elem.bounding_box()
    if raw_bbox is None:
        return action
    if raw_bbox["width"] * raw_bbox["height"] > _PHANTOM_MIN_AREA:
        return action  # not a phantom

    resolved_bbox, _, parent_bid = _resolve_clickable_bbox(elem)
    if resolved_bbox is None:
        return action

    if parent_bid:
        return action.replace(f"'{bid}'", f"'{parent_bid}'", 1)

    cx = resolved_bbox["x"] + resolved_bbox["width"] / 2
    cy = resolved_bbox["y"] + resolved_bbox["height"] / 2
    return f"{_PHANTOM_VERB_MAP[verb]}({cx:.1f}, {cy:.1f})"
