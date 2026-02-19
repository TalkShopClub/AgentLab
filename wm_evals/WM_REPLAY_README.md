# World Model Replay - README

This system allows you to replay world model predictions on existing trajectory steps to compare how agents make decisions with and without world model context.

## Overview

The workflow consists of three steps:

1. **Create Input JSON**: List of trajectory paths and step IDs to replay
2. **Run Replay Script**: Generates world model predictions for each step
3. **Visualize Results**: Creates HTML comparison view

## Files

- `replay_wm_predictions.py` - Main script to replay WM predictions
- `visualize_replay_results.py` - Creates HTML comparison visualization
- `example_trajectories.json` - Example input format

## Usage

### Step 1: Create Input JSON

Create a JSON file with trajectory paths and step IDs:

```json
[
    {
        "traj_path": "/path/to/trajectory_folder_1",
        "step_id": 5
    },
    {
        "traj_path": "/path/to/trajectory_folder_2",
        "step_id": 10
    }
]
```

### Step 2: Run Replay Script

Make sure the Emu3.5 world model server is running, then:

```bash
cd /Users/lakshyagupta/skyfall/ewm_benchmarks/AgentLab
source .venv/bin/activate

cd wm_evals
python replay_wm_predictions.py \
    --input example_trajectories.json \
    --output-dir wm_replay_results/ \
    --wm-mode text \
    --level l3
```

**Options:**
- `--input, -i`: JSON file with trajectory entries (required)
- `--output-dir, -o`: Directory to save results (required)
- `--wm-server-url`: World model server URL (default: https://z66y0a4p8qruii-8000.proxy.runpod.net/)
- `--wm-mode`: Prediction mode - `text` or `image` (default: text)
- `--level`: Task level - `l2` or `l3` (default: l3, affects goal extraction)
- `--model`: LLM model for candidate generation (default: openai/gpt-5-2025-08-07)

### Step 3: Create Visualization

```bash
python visualize_replay_results.py \
    --input wm_replay_results/ \
    --output comparison.html \
    --wm-mode text
```

**Options:**
- `--input, -i`: Directory with replay results (required)
- `--output, -o`: Output HTML file (default: replay_comparison.html)
- `--wm-mode`: Display mode for predictions (default: text)

Open the HTML file in your browser to view the comparison.

## Output Format

The replay script creates:

1. **Individual JSON files**: One per trajectory step
   - Format: `{traj_name}_step_{step_id}.json`
   - Contains: candidates, predictions, original action/thought

2. **Combined JSON**: `all_predictions.json`
   - All results in a single file

3. **HTML Visualization**: Side-by-side comparison showing:
   - Left: Original agent decision (no world model)
   - Right: World model predictions with all candidates

## What Gets Compared

### Original Agent (Left Side)
- **Action Taken**: The actual action the agent chose
- **Agent's Reasoning**: The thought/reasoning behind the action

### With World Model (Right Side)
- **Candidate Actions**: Top 5 candidate actions generated
- **Action Descriptions**: Plain English description of each action
- **Rationales**: Why each candidate might be good
- **Predictions**: World model's prediction of next state for each candidate
  - Text mode: Natural language description
  - Image mode: Predicted screenshot

## How It Works

1. **Load Trajectory**: Reads the trajectory up to the specified step
2. **Reconstruct Context**: Builds the same context the agent had (goal, history)
3. **Generate Candidates**: Uses the same prompts to generate 5 candidate actions
4. **Get Predictions**: Calls Emu3.5 to predict next state for each candidate
5. **Compare**: Shows original decision vs. what the agent would see with WM context

## Example Workflow

```bash
# 1. Activate environment
cd /Users/lakshyagupta/skyfall/ewm_benchmarks/AgentLab
source .venv/bin/activate
cd wm_evals

# 2. Make sure WM server is running
# (Check health at https://z66y0a4p8qruii-8000.proxy.runpod.net/health)

# 3. Run replay on example
python replay_wm_predictions.py \
    -i example_trajectories.json \
    -o wm_replay_results/ \
    --wm-mode text

# 4. Create visualization
python visualize_replay_results.py \
    -i wm_replay_results/ \
    -o comparison.html

# 5. Open in browser
open comparison.html
```

## Notes

- The original trajectories must have step pickle files (`step_*.pkl.gz`)
- The Emu3.5 server must be running and accessible
- Text mode is faster than image mode
- Each prediction can take 1-2 minutes depending on the server
- Results are saved incrementally, so you can stop and resume

## Troubleshooting

**"World model server not responding"**
- Check if the server is running
- Verify the URL is correct
- Test with: `curl https://z66y0a4p8qruii-8000.proxy.runpod.net/health`

**"step_id out of range"**
- Check that the trajectory has enough steps
- Step IDs are 0-indexed

**"No module named 'agentlab'"**
- Make sure you activated the virtual environment
- Run: `source .venv/bin/activate`
