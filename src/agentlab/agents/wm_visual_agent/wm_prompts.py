"""Prompt templates for the world-model-augmented visual agent.

Phase 1: Ask the agent to propose top-5 candidate actions (code + text description + rationale).
Phase 2: Present world model predictions and ask the agent to select the best action.
"""

import re

from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.visual_agent.visual_agent_prompts import (
    History,
    Observation,
    PromptFlags,
    make_instructions,
)
from agentlab.llm.llm_utils import HumanMessage, image_to_jpg_base64_url, parse_html_tags_raise


class CandidateGenerationPrompt:
    """Phase 1: Generate top-5 candidate actions with code, text description, and rationale."""

    def __init__(
        self,
        action_set: AbstractActionSet,
        obs: dict,
        actions: list[str],
        thoughts: list[str],
        flags: PromptFlags,
        model_name: str = "",
    ):
        self.flags = flags
        self.instructions = make_instructions(obs, flags.enable_chat, flags.extra_instructions)
        self.obs = Observation(obs, flags.obs)
        self.history = History(actions, thoughts)
        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)

    @property
    def prompt(self) -> HumanMessage:
        msg = HumanMessage(self.instructions.prompt)
        msg.add_text(
            f"""\
            {self.obs.prompt}\
            {self.history.prompt}\
            {self.action_prompt.prompt}\

            # Your Task
            Propose exactly 5 candidate actions you think are most promising right now.
            For each candidate, provide three things:
            - <action>: The exact action code from the action set (e.g. click('70'))
            - <action_text>: A plain-English description of what the action does (e.g. "Click the quantity dropdown button")
            - <rationale>: Why you think this action would help achieve the goal

            Use this exact format:

            <candidates>
            <candidate_1>
            <action>click('70')</action>
            <action_text>Click the quantity dropdown button</action_text>
            <rationale>Need to select the number of items</rationale>
            </candidate_1>
            <candidate_2>
            <action>fill('42', 'laptop')</action>
            <action_text>Type "laptop" into the search field</action_text>
            <rationale>Need to search for the product</rationale>
            </candidate_2>
            <candidate_3>
            <action>...</action>
            <action_text>...</action_text>
            <rationale>...</rationale>
            </candidate_3>
            <candidate_4>
            <action>...</action>
            <action_text>...</action_text>
            <rationale>...</rationale>
            </candidate_4>
            <candidate_5>
            <action>...</action>
            <action_text>...</action_text>
            <rationale>...</rationale>
            </candidate_5>
            </candidates>
            """
        )
        return self.obs.add_screenshot(msg)

    def parse_candidates(self, text: str) -> list[dict]:
        """Extract candidate actions from the LLM response.

        Returns list of dicts with keys: action, action_text, rationale
        """
        candidates = []
        for i in range(1, 6):
            block = re.search(
                rf"<candidate_{i}>(.*?)</candidate_{i}>", text, re.DOTALL
            )
            if not block:
                continue
            content = block.group(1)
            action_m = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            text_m = re.search(r"<action_text>(.*?)</action_text>", content, re.DOTALL)
            rat_m = re.search(r"<rationale>(.*?)</rationale>", content, re.DOTALL)
            candidates.append({
                "action": action_m.group(1).strip() if action_m else "",
                "action_text": text_m.group(1).strip() if text_m else "",
                "rationale": rat_m.group(1).strip() if rat_m else "",
            })
        return candidates


class InformedSelectionPrompt:
    """Phase 2: Present predictions, ask agent to pick the best action."""

    def __init__(
        self,
        action_set: AbstractActionSet,
        obs: dict,
        candidates: list[dict],
        predictions: list[dict],
        flags: PromptFlags,
        mode: str = "image",
        model_name: str = "",
    ):
        self.flags = flags
        self.instructions = make_instructions(obs, flags.enable_chat, flags.extra_instructions)
        self.obs = Observation(obs, flags.obs)
        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)
        self.candidates = candidates
        self.predictions = predictions
        self.mode = mode

    @property
    def prompt(self) -> HumanMessage:
        msg = HumanMessage(self.instructions.prompt)
        msg.add_text(self.obs.prompt)
        msg = self.obs.add_screenshot(msg)

        msg.add_text(
            "\n# World Model Predictions\n"
            "A world model has predicted the next state of the webpage for each candidate action. "
            "Use these predictions to select the best action.\n"
        )

        for i, (cand, pred) in enumerate(zip(self.candidates, self.predictions), 1):
            msg.add_text(
                f"\n## Candidate {i}\n"
                f"- Action: `{cand['action']}`\n"
                f"- Description: {cand['action_text']}\n"
            )

            if self.mode == "image" and pred.get("image") is not None:
                msg.add_text(f"Predicted next state for Candidate {i}:")
                img_url = image_to_jpg_base64_url(pred["image"])
                msg.add_image(img_url, detail=self.flags.obs.openai_vision_detail)
            elif self.mode == "text" and pred.get("text"):
                msg.add_text(f"Predicted next state for Candidate {i}: {pred['text']}")
            else:
                msg.add_text(f"Predicted next state for Candidate {i}: [unavailable]")

        msg.add_text(
            """
            # Your Task
            Based on the current observation and the world model predictions above, select the single best action.
            Explain your reasoning, then provide the chosen action code exactly as it appeared in the candidates.

            <think>
            Your reasoning about which predicted next state best advances the goal.
            </think>
            <action>
            the_action_code
            </action>
            """
        )
        return msg

    def parse_answer(self, text: str) -> dict:
        ans = parse_html_tags_raise(text, keys=["action"], optional_keys=["think"], merge_multiple=True)
        return ans