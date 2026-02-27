"""Prompt templates for the world-model-augmented visual agent.

Phase 1: Ask the agent to propose top-5 candidate actions (code + text description + rationale).
Phase 2: Present world model predictions and ask the agent to select the best action.
"""

import re

from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.visual_agent.visual_agent_prompts import (
    History,
    PromptFlags,
    make_instructions,
)
from agentlab.llm.llm_utils import HumanMessage, image_to_jpg_base64_url, parse_html_tags_raise


class CandidateGenerationPrompt:
    """Phase 1: Generate top-K candidate actions with code, text description, and rationale."""

    def __init__(
        self,
        action_set: AbstractActionSet,
        obs: dict,
        actions: list[str],
        thoughts: list[str],
        flags: PromptFlags,
        model_name: str = "",
        n_candidates: int = 5,
    ):
        self.flags = flags
        self.n_candidates = n_candidates
        self.instructions = make_instructions(obs, flags.enable_chat, flags.extra_instructions)
        self.obs = dp.Observation(obs, flags.obs)
        self.history = History(actions, thoughts)
        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)

    def _candidate_examples(self) -> str:
        lines = ["<candidates>"]
        for i in range(1, self.n_candidates + 1):
            if i == 1:
                action, text, rationale = "click('70')", "Click the quantity dropdown button", "Need to select the number of items"
            elif i == 2:
                action, text, rationale = "fill('42', 'laptop')", 'Type "laptop" into the search field', "Need to search for the product"
            else:
                action, text, rationale = "...", "...", "..."
            lines.append(f"<candidate_{i}>")
            lines.append(f"<rationale>{rationale}</rationale>")
            lines.append(f"<action>{action}</action>")
            lines.append(f"<action_text>{text}</action_text>")
            lines.append(f"</candidate_{i}>")
        lines.append("</candidates>")
        return "\n".join(lines)

    @property
    def prompt(self) -> HumanMessage:
        msg = HumanMessage(self.instructions.prompt)
        msg.add_text(
            f"""\
            {self.obs.prompt}\
            {self.history.prompt}\
            {self.action_prompt.prompt}\

            # Your Task
            Propose exactly {self.n_candidates} candidate actions you think are most promising right now.
            For each candidate, provide three things in this order:
            - <rationale>: Why you think this action would help achieve the goal (write this first)
            - <action>: The exact action code from the action set (e.g. click('70'))
            - <action_text>: A plain-English description of what the action does (e.g. "Click the quantity dropdown button")

            ## Diversity Guidelines
            Your candidates will be executed in the real environment to observe their effects, so **maximize the
            information gained** by proposing diverse actions:
            - **Target different elements**: Spread candidates across distinct interactive elements (different BIDs).
              Avoid proposing multiple actions on the same element unless you are highly confident it is the only
              viable path forward.
            - Prioritize bid based actions over coordinate based actions in the diverse sampling. If there are enough bid based potential candidates, do not propose coordinate based actions.
            - **Explore alternative strategies**: If you are unsure of the correct path, use candidates to test
              fundamentally different approaches (e.g., using a menu vs. a search bar vs. a direct link) rather than
              minor variations of the same approach.
            - **Double-check BID grounding**: Before writing each action, verify the BID you are referencing actually
              corresponds to the element you intend to interact with. A common failure mode is picking a nearby BID
              that belongs to a different element.

            Use this exact format:

            {self._candidate_examples()}
            """
        )
        return self.obs.add_screenshot(msg)

    def parse_candidates(self, text: str) -> list[dict]:
        """Extract candidate actions from the LLM response.

        Returns list of dicts with keys: action, action_text, rationale
        """
        candidates = []
        for i in range(1, self.n_candidates + 1):
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
        self.obs = dp.Observation(obs, flags.obs)
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