#!/usr/bin/env python3
"""
Create HTML visualization from replay_wm_predictions.py results.
Shows side-by-side comparison of original agent decision vs world model predictions.

Usage:
    python visualize_replay_results.py --input wm_replay_results/ --output replay_viz.html
"""

import argparse
import base64
import html
import json
from pathlib import Path


def create_html_visualization(results_dir: Path, output_html: Path, wm_mode: str = "text"):
    """Create HTML visualization from replay results."""

    # Load all prediction results
    all_results_file = results_dir / "all_predictions.json"
    if not all_results_file.exists():
        print(f"Error: {all_results_file} not found")
        return False

    with open(all_results_file, "r") as f:
        all_results = json.load(f)

    if not all_results:
        print("No results to visualize")
        return False

    html_parts = []

    # HTML header
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>World Model Replay Results - Comparison</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .header-info {
            font-size: 14px;
            color: #666;
        }
        .result-entry {
            background: white;
            margin-bottom: 30px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .entry-header {
            background: #007bff;
            color: white;
            padding: 15px 20px;
        }
        .entry-header h2 {
            margin: 0;
            font-size: 18px;
        }
        .entry-info {
            font-size: 14px;
            opacity: 0.9;
            margin-top: 5px;
        }
        .entry-content {
            padding: 20px;
        }
        .goal-section {
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 20px;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }
        .goal-section strong {
            color: #007bff;
        }
        .screenshot-section {
            margin-bottom: 20px;
        }
        .screenshot-label {
            font-size: 14px;
            font-weight: 600;
            color: #666;
            margin-bottom: 8px;
        }
        .screenshot {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .comparison-section {
            display: grid;
            grid-template-columns: 1fr 1.5fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .comparison-box {
            border: 2px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
        }
        .comparison-header {
            padding: 12px 15px;
            font-weight: 600;
            font-size: 15px;
        }
        .original-header {
            background: #ffc107;
            color: #000;
        }
        .wm-header {
            background: #17a2b8;
            color: #fff;
        }
        .selected-header {
            background: #28a745;
            color: #fff;
        }
        .comparison-content {
            padding: 15px;
        }
        .action-box {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            border-left: 4px solid #28a745;
        }
        .action-box strong {
            color: #28a745;
            display: block;
            margin-bottom: 6px;
        }
        .action-text {
            font-family: monospace;
            font-size: 13px;
            color: #212529;
        }
        .thought-box {
            background: #fff3cd;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            border-left: 4px solid #ffc107;
        }
        .thought-box strong {
            color: #856404;
            display: block;
            margin-bottom: 6px;
        }
        .thought-text {
            font-size: 13px;
            color: #856404;
            white-space: pre-wrap;
        }
        .wm-section {
            background: #f0f8ff;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            border-left: 4px solid #17a2b8;
        }
        .wm-section h3 {
            margin: 0 0 15px 0;
            color: #17a2b8;
            font-size: 16px;
        }
        .wm-candidate {
            background: white;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }
        .wm-candidate-header {
            font-weight: 600;
            color: #495057;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid #dee2e6;
        }
        .wm-candidate-body {
            font-size: 13px;
        }
        .wm-field {
            margin: 6px 0;
            color: #212529;
        }
        .wm-prediction {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #dee2e6;
        }
        .wm-pred-img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 8px;
        }
        .wm-pred-text {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin-top: 8px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
        }
        .no-data {
            color: #6c757d;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>World Model Replay Results - Comparison View</h1>
        <div class="header-info">
            Total Results: """ + str(len(all_results)) + """<br>
            Mode: """ + wm_mode.upper() + """ predictions
        </div>
    </div>
""")

    # Process each result
    for idx, result in enumerate(all_results, 1):
        task_name = html.escape(result.get("task_name", "Unknown"))
        traj_path = html.escape(result.get("traj_path", ""))
        step_id = result.get("step_id", 0)
        goal = html.escape(result.get("goal", "")).replace("\n", "<br>")

        original_action = result.get("original_action")
        original_thought = result.get("original_thought")

        html_parts.append(f"""
    <div class="result-entry">
        <div class="entry-header">
            <h2>Result #{idx}: {task_name}</h2>
            <div class="entry-info">Step ID: {step_id} | Path: {traj_path}</div>
        </div>
        <div class="entry-content">
            <div class="goal-section">
                <strong>Task Goal:</strong><br>
                {goal}
            </div>

            <div class="screenshot-section">
                <div class="screenshot-label">Current State Screenshot (SOM)</div>
                <img src="data:image/png;base64,{result['screenshot_som_b64']}" class="screenshot">
            </div>

            <div class="comparison-section">
                <div class="comparison-box">
                    <div class="comparison-header original-header">
                        📋 ORIGINAL AGENT DECISION (No World Model)
                    </div>
                    <div class="comparison-content">
""")

        # Original action
        if original_action:
            action_escaped = html.escape(str(original_action)).replace("\n", "<br>")
            html_parts.append(f"""
                        <div class="action-box">
                            <strong>Action Taken:</strong>
                            <div class="action-text">{action_escaped}</div>
                        </div>
""")
        else:
            html_parts.append("""
                        <div class="no-data">No action recorded</div>
""")

        # Original thought
        if original_thought:
            thought_escaped = html.escape(str(original_thought))
            html_parts.append(f"""
                        <div class="thought-box">
                            <strong>Agent's Reasoning:</strong>
                            <div class="thought-text">{thought_escaped}</div>
                        </div>
""")
        else:
            html_parts.append("""
                        <div class="no-data">No thought recorded</div>
""")

        html_parts.append("""
                    </div>
                </div>

                <div class="comparison-box">
                    <div class="comparison-header wm-header">
                        🔮 WITH WORLD MODEL PREDICTIONS
                    </div>
                    <div class="comparison-content">
                        <div class="wm-section">
                            <h3>Generated Candidates & Predictions</h3>
""")

        candidates = result.get("candidates", [])
        predictions = result.get("predictions", [])

        for j, (cand, pred) in enumerate(zip(candidates, predictions), 1):
            cand_action = html.escape(str(cand.get("action", ""))).replace("\n", "<br>")
            cand_text = html.escape(str(cand.get("action_text", ""))).replace("\n", "<br>")
            cand_rationale = html.escape(str(cand.get("rationale", ""))).replace("\n", "<br>")

            html_parts.append(f"""
                            <div class="wm-candidate">
                                <div class="wm-candidate-header">Candidate #{j}</div>
                                <div class="wm-candidate-body">
                                    <div class="wm-field"><strong>Action Code:</strong> {cand_action}</div>
                                    <div class="wm-field"><strong>Description:</strong> {cand_text}</div>
                                    <div class="wm-field"><strong>Rationale:</strong> {cand_rationale}</div>
""")

            if pred.get("image"):
                pred_img_b64 = pred["image"]
                html_parts.append(f"""
                                    <div class="wm-prediction">
                                        <strong>🖼️ Predicted Next State:</strong><br>
                                        <img src="data:image/png;base64,{pred_img_b64}" class="wm-pred-img">
                                    </div>
""")
            elif pred.get("text"):
                pred_text = html.escape(str(pred["text"]))
                html_parts.append(f"""
                                    <div class="wm-prediction">
                                        <strong>📝 Predicted Next State:</strong>
                                        <div class="wm-pred-text">{pred_text}</div>
                                    </div>
""")

            html_parts.append("""
                                </div>
                            </div>
""")

        html_parts.append("""
                        </div>
                    </div>
                </div>

                <div class="comparison-box">
                    <div class="comparison-header selected-header">
                        ✅ AGENT'S SELECTION (After Seeing WM)
                    </div>
                    <div class="comparison-content">
""")

        # Agent selection after seeing WM
        agent_selection = result.get("agent_selection", {})
        selected_action = agent_selection.get("selected_action")
        selection_thought = agent_selection.get("selection_thought")

        if selected_action:
            action_escaped = html.escape(str(selected_action)).replace("\n", "<br>")
            html_parts.append(f"""
                        <div class="action-box">
                            <strong>Selected Action:</strong>
                            <div class="action-text">{action_escaped}</div>
                        </div>
""")
        else:
            html_parts.append("""
                        <div class="no-data">No action selected</div>
""")

        if selection_thought:
            thought_escaped = html.escape(str(selection_thought))
            html_parts.append(f"""
                        <div class="thought-box">
                            <strong>Agent's Reasoning (After WM):</strong>
                            <div class="thought-text">{thought_escaped}</div>
                        </div>
""")
        else:
            html_parts.append("""
                        <div class="no-data">No thought recorded</div>
""")

        html_parts.append("""
                    </div>
                </div>
            </div>
        </div>
    </div>
""")

    html_parts.append("""
</body>
</html>""")

    # Write HTML file
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"HTML visualization created: {output_html}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create HTML visualization comparing original agent decisions vs world model predictions"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Directory containing replay results (with all_predictions.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="replay_comparison.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--wm-mode",
        type=str,
        default="text",
        choices=["text", "image"],
        help="World model mode (for display purposes)",
    )

    args = parser.parse_args()

    results_dir = Path(args.input)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    output_html = Path(args.output)

    success = create_html_visualization(results_dir, output_html, args.wm_mode)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
