# oracle_wm

Oracle world-model pipeline for WorkArena. Instead of letting an agent act blindly, the oracle pipeline grounds each decision in real browser execution: the LLM proposes K candidate actions, each is executed in an isolated environment replay to capture its true outcome, and the LLM then selects the best one from the actual resulting screenshots. This produces near-optimal trajectories that serve as upper-bound baselines and training data.

---

## How the pipeline works

Each step of the episode runs three phases:

**Phase 1 — Candidate generation**
The LLM receives the current observation (SOM screenshot or AXTree depending on mode) and produces K candidate actions with rationales.

**Phase 2a — Candidate exploration**
Each candidate is executed in a fresh environment that has been replayed to the current decision point. The resulting screenshot and SOM are saved. Environments are run sequentially to avoid data conflicts from parallel resets of the same deterministic task seed.

**Phase 2a.5 — Effect description**
The LLM compares the before/after screenshots for each candidate and writes a plain-text description of what changed visually.

**Phase 2b — Selection**
A final environment replay returns to the decision point. The LLM is shown all K candidate outcomes (text descriptions and/or screenshots) and picks the best one. If it judges all candidates unsatisfactory it can request a resample, which re-runs Phase 1 with feedback about why the first batch was rejected.

**Execution**
The selected action is executed on the live environment. Artifacts are checkpointed so the run can be resumed from any step.

### BID translation

BIDs (BrowserGym element identifiers) change between environment instances. Before each candidate is executed and before the selected action is committed, BIDs are translated using fingerprint matching: each element is fingerprinted by its ARIA role, name, ancestor path, and sibling context. This keeps actions valid across fresh environment replays.

### Zero-size element resolution

ServiceNow renders some accessible elements (e.g. comboboxes) as 1×1 px DOM anchors. Playwright clicks the geometric center of these, which lands on a dead pixel. Before any `click`, `dblclick`, `right_click`, or `hover` action is executed, the pipeline detects zero-size BIDs and transparently rewrites the action to target the resolved visual container. This is implemented in `agentlab/utils/phantom_actions.py` and runs automatically in both the oracle pipeline and the standard agent loop.

---

## Directory structure

```
oracle_wm/
├── oracle_pipeline/
│   ├── oracle_loop.py          # Core pipeline: all phases, replay logic, commit loop
│   ├── oracle_prompts.py       # Prompt classes for effect description and candidate selection
│   └── run_oracle.py           # CLI entry point
│
├── oracle_html.py              # Generates interactive HTML debug report from a run directory
├── repl.py                     # Interactive REPL for manual action testing against a live env
│
├── _bid_utils.py               # Shared utilities: env creation, BID mapping, action translation
├── _bid_replay.py              # Replay and translate-replay modes for BID analysis
├── _bid_compare.py             # Compare BID maps across trajectories or original vs replay
├── _bid_html.py                # HTML rendering for BID comparison output
├── bid_analysis.py             # CLI tool for replaying trajectories and comparing BID stability
├── task_seed_experiment.py     # Verifies that task randomness is stable across seeds
│
├── bid_snapshots/              # Output directory for bid_analysis.py
└── runs/                       # Per-run artifact directory (created automatically)
```

---

## Running the oracle pipeline

```bash
python -m oracle_wm.oracle_pipeline.run_oracle \
    --task workarena.servicenow.basic-expense-management-medium-l2 \
    --seed 42
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | required | BrowserGym task name |
| `--seed` | required | Task seed (integer) |
| `--model` | `openai/gpt-5-2025-08-07` | LLM model key from `CHAT_MODEL_ARGS_DICT` |
| `--n-candidates` | `5` | Number of candidate actions generated per step |
| `--max-steps` | `30` | Maximum steps before the episode is truncated |
| `--save-dir` | `oracle_results` | Directory for final `StepInfo` results |
| `--run-dir` | `oracle_wm/runs` | Directory for per-step debug artifacts |
| `--agent-mode` | `vision` | `vision`: SOM screenshot for generation; `text`: AXTree only |
| `--headless` | off | Run the browser headless |
| `--resume-from N` | `0` | Resume from step N, reading committed history from the run dir |
| `--cleanup` | off | Delete orphaned `@workarena.com` users before starting (use after interrupted runs) |
| `--no-sel-effects` | off | Exclude effect text descriptions from the selection prompt |
| `--no-sel-images` | off | Exclude candidate screenshots from the selection prompt |

### Examples

Vision mode with 3 candidates:
```bash
python -m oracle_wm.oracle_pipeline.run_oracle \
    --task workarena.servicenow.basic-expense-management-medium-l2 \
    --seed 42 --n-candidates 3 --agent-mode vision
```

Text-only mode (AXTree, no screenshots):
```bash
python -m oracle_wm.oracle_pipeline.run_oracle \
    --task workarena.servicenow.basic-expense-management-medium-l2 \
    --seed 42 --agent-mode text
```

Resume from step 5 after an interrupted run:
```bash
python -m oracle_wm.oracle_pipeline.run_oracle \
    --task workarena.servicenow.basic-expense-management-medium-l2 \
    --seed 42 --resume-from 5
```

Selection with text descriptions only (no candidate images):
```bash
python -m oracle_wm.oracle_pipeline.run_oracle \
    --task workarena.servicenow.basic-expense-management-medium-l2 \
    --seed 42 --no-sel-images
```

Clean up orphaned users from a previous interrupted run, then start fresh:
```bash
python -m oracle_wm.oracle_pipeline.run_oracle \
    --task workarena.servicenow.basic-expense-management-medium-l2 \
    --seed 42 --cleanup
```

---

## Run artifacts

Each run writes to `oracle_wm/runs/<task>_seed<N>/`:

```
goal.txt
step_0/
    selection.json              # selected index, thought, bid translation note
    committed_step.json         # action + BID entry used for future replays
    decision_bid_map.json       # full BID map at decision point
    initial/
        current.png             # screenshot of env state before generation
        current_som.png
        decision_point_som.png  # SOM at the moment of selection
        candidates.json         # K candidate actions with rationales
        candidate_effects.json  # LLM-written effect description per candidate
        bid_translations.json   # BID translation notes per candidate
        phase1_prompt.txt       # full Phase 1 prompt (text only, images as [IMAGE])
        phase1_response.txt
        phase2_prompt.txt
        phase2_response.txt
        future_1.png            # screenshot after candidate 1
        future_1_som.png
        future_2.png ...
    resample/                   # only present if the LLM requested a resample
        (same structure as initial/)
step_1/
    ...
```

Final `StepInfo` results (compatible with standard AgentLab result loading) are written to `oracle_results/<task>_seed<N>/`.

---

## Generating the HTML debug report

After a run completes (or mid-run), generate an interactive HTML report:

```bash
python oracle_wm/oracle_html.py \
    --run-dir oracle_wm/runs/workarena.servicenow.basic-expense-management-medium-l2_seed42
```

Or by task and seed (resolves the run dir automatically):
```bash
python oracle_wm/oracle_html.py \
    --task workarena.servicenow.basic-expense-management-medium-l2 --seed 42
```

The report shows each step as a card with: current state, all K candidate outcomes side by side, effect descriptions, BID translation notes, the selection reasoning, and whether a resample was triggered.

---

## Interactive REPL

For manual debugging against a live ServiceNow environment:

```bash
python oracle_wm/repl.py \
    --task workarena.servicenow.basic-expense-management-medium-l2 \
    --seed 30
```

Type actions at the prompt. After each step the AXTree, SOM, and screenshot in `oracle_wm/repl_out/` are overwritten so you can watch them update in an editor side by side.

Special commands (do not step the environment):
- `coord('a60')` — prints the element's bounding box and center, resolves zero-size anchors to their visual container, saves an annotated screenshot to `repl_out/coord_a60.png`

The output directory is wiped clean on each REPL startup.

---

## BID analysis tools

These scripts were used to verify BID stability across ServiceNow pool instances and are not needed for normal oracle runs.

**Replay a trajectory on a fresh env instance:**
```bash
python oracle_wm/bid_analysis.py replay \
    results2/<study>/<task_dir> --output replay1.json
```

**Replay with BID translation (fingerprint-matched):**
```bash
python oracle_wm/bid_analysis.py translate-replay \
    results2/<study>/<task_dir> --output translated1.json
```

**Compare two replay outputs step by step:**
```bash
python oracle_wm/bid_analysis.py compare replay1.json replay2.json
```

**Compare original vs replayed BIDs within one file:**
```bash
python oracle_wm/bid_analysis.py self-compare replay1.json
```

**Generate side-by-side HTML comparison:**
```bash
python oracle_wm/bid_analysis.py html translated1.json
```

**Verify task determinism across seeds:**
```bash
python oracle_wm/task_seed_experiment.py \
    --task workarena.servicenow.dashboard-retrieve-catalog-and-max-order-apple-watch-l2 \
    --seed 42 --n 3
```