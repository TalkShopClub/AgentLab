"""Compare BID maps across trajectories or between original and replay."""

import json
import os
import re
import tempfile
from pathlib import Path


def compare_trajectories(path1: str, path2: str) -> None:
    """
    Compare BID maps from two replay JSONs step by step.
    Accepts either the 'replayed_steps' field or the 'original_steps' field of the JSON.
    Also accepts 'translated_steps'.

    For cross-instance comparison: compare replayed_steps of replay1 vs replay2.
    For original-vs-replay:       compare original_steps of replay1 vs replayed_steps of replay1.
    """
    d1 = json.loads(Path(path1).read_text())
    d2 = json.loads(Path(path2).read_text())

    steps1 = d1.get("replayed_steps") or d1.get("translated_steps") or d1.get("original_steps") or []
    steps2 = d2.get("replayed_steps") or d2.get("translated_steps") or d2.get("original_steps") or []

    print(f"File1: {path1}  ({len(steps1)} steps)")
    print(f"File2: {path2}  ({len(steps2)} steps)")
    n = min(len(steps1), len(steps2))

    total_stable, total_drift, total_miss = 0, 0, 0
    for i in range(n):
        s1, s2 = steps1[i], steps2[i]
        fp1 = {v["fingerprint"]: bid for bid, v in s1["bid_map"].items()}
        fp2 = {v["fingerprint"]: bid for bid, v in s2["bid_map"].items()}

        stable, drift, miss2, miss1 = [], [], [], []
        for fp in set(fp1) | set(fp2):
            if fp.startswith("AMBIGUOUS::"):
                continue
            b1, b2 = fp1.get(fp), fp2.get(fp)
            if b1 and b2:
                (stable if b1 == b2 else drift).append((fp, b1, b2))
            elif b1:
                miss2.append((fp, b1))
            else:
                miss1.append((fp, b2))

        total = len(stable) + len(drift) + len(miss1) + len(miss2)
        pct = lambda x: f"{x / max(total, 1) * 100:.1f}%"
        total_stable += len(stable)
        total_drift += len(drift)
        total_miss += len(miss1) + len(miss2)

        url1, url2 = s1.get("url", ""), s2.get("url", "")
        action_err = s2.get("action_error")
        url_match = "" if url1 == url2 else f"  [URL MISMATCH: {url1} vs {url2}]"
        err_note = f"  [ACTION FAILED: {action_err}]" if action_err else ""
        print(f"\nStep {i}: {s2.get('action', 'reset')!r:.60}{url_match}{err_note}")
        print(f"  total={total}  stable={len(stable)} ({pct(len(stable))})  "
              f"drift={len(drift)} ({pct(len(drift))})  "
              f"miss_f2={len(miss2)}  miss_f1={len(miss1)}")

        action = s1.get("action") or ""
        m = re.match(r"\w+\('([^']+)'", action)
        if m:
            acting_bid = m.group(1)
            acting_entry = s1["bid_map"].get(acting_bid)
            if acting_entry:
                fp = acting_entry["fingerprint"]
                replay_bid = fp2.get(fp)
                status = "STABLE" if acting_bid == replay_bid else ("DRIFT" if replay_bid else "MISSING_IN_F2")
                print(f"  ACTION  f1_bid={acting_bid!r}  fp={fp!r}  ->  f2_bid={replay_bid!r}  [{status}]")
            else:
                print(f"  ACTION  f1_bid={acting_bid!r}  [NOT IN F1 BID MAP]")

        for fp, b1, b2 in sorted(drift)[:10]:
            print(f"    DRIFT  {fp!r:<50}  f1={b1}  f2={b2}")

    all_steps = total_stable + total_drift + total_miss
    print(f"\n{'='*60}")
    print(f"TOTAL across {n} steps: "
          f"stable={total_stable} ({total_stable/max(all_steps,1)*100:.1f}%)  "
          f"drift={total_drift} ({total_drift/max(all_steps,1)*100:.1f}%)  "
          f"missing={total_miss}")


def compare_original_vs_replay(replay_path: str) -> None:
    """
    Within a single replay JSON, compare the original trajectory's BID maps against
    the replayed (or translated) BID maps to directly show cross-instance BID drift.
    """
    d = json.loads(Path(replay_path).read_text())
    orig = d.get("original_steps", [])
    repl = d.get("replayed_steps") or d.get("translated_steps") or []
    if not orig or not repl:
        print("Missing original_steps and replayed_steps/translated_steps in the file.")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
        json.dump({"replayed_steps": orig}, f1)
        tmp1 = f1.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
        json.dump({"replayed_steps": repl}, f2)
        tmp2 = f2.name
    try:
        compare_trajectories(tmp1, tmp2)
    finally:
        os.unlink(tmp1)
        os.unlink(tmp2)
