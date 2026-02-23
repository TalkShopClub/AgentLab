"""Oracle selection prompt: present K real environment screenshots and ask agent to pick one."""

import re

from agentlab.agents.visual_agent.visual_agent_prompts import (
    History,
    Observation,
    PromptFlags,
    make_instructions,
)
from agentlab.llm.llm_utils import HumanMessage, image_to_jpg_base64_url


class OracleSelectionPrompt:
    """Phase 2 (oracle): show K real screenshots from environment execution, ask agent to pick."""

    def __init__(
        self,
        obs: dict,
        actions: list[str],
        thoughts: list[str],
        candidates: list[tuple[str, object]],  # [(action_text, screenshot_array), ...]
        flags: PromptFlags,
    ):
        self.instructions = make_instructions(obs, flags.enable_chat, flags.extra_instructions)
        self.obs = Observation(obs, flags.obs)
        self.history = History(actions, thoughts)
        self.candidates = candidates
        self.flags = flags

    @property
    def prompt(self) -> HumanMessage:
        msg = HumanMessage(self.instructions.prompt)
        msg.add_text(f"{self.obs.prompt}{self.history.prompt}")
        msg = self.obs.add_screenshot(msg)

        msg.add_text(
            "\n# Oracle Candidate Actions\n"
            "The following candidate actions were each executed in the real environment. "
            "You are shown the resulting screenshot for each.\n"
        )

        for i, (action_text, screenshot) in enumerate(self.candidates, 1):
            msg.add_text(f"\n## Candidate {i}\nAction: `{action_text}`\nResulting state:")
            img_url = image_to_jpg_base64_url(screenshot)
            msg.add_image(img_url, detail=self.flags.obs.openai_vision_detail)

        msg.add_text(
            f"\n# Your Task\n"
            f"Select the single best candidate (1–{len(self.candidates)}) based on which resulting "
            f"state best advances the goal.You must generate the thought in the <think> tag before selecting the candidate using the <selection> tag. \n\n"
            f"<think>\nYour reasoning about which candidate best advances the goal.\n</think>\n"
            f"<selection>N (between 1 and {len(self.candidates)})</selection>"
        )
        return msg

    def parse_answer(self, text: str) -> tuple[int, str]:
        """Returns (0-indexed selection, reasoning)."""
        think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        sel_m = re.search(r"<selection>\s*(\d+)\s*</selection>", text, re.DOTALL)
        reasoning = think_m.group(1).strip() if think_m else ""
        if not sel_m:
            raise ValueError(f"No <selection> tag found in response: {text[:200]}")
        idx = int(sel_m.group(1)) - 1  # convert 1-indexed to 0-indexed
        if idx < 0 or idx >= len(self.candidates):
            raise ValueError(f"Selection {idx + 1} out of range (1–{len(self.candidates)})")
        return idx, reasoning
