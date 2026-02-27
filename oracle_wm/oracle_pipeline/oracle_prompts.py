"""Oracle selection prompt: present K real environment screenshots and ask agent to pick one."""

import re

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.visual_agent.visual_agent_prompts import (
	History,
	PromptFlags,
	make_instructions,
)
from agentlab.llm.llm_utils import HumanMessage, image_to_jpg_base64_url


class CandidateEffectDescriptionPrompt:
	"""Describe what visually changed after each candidate action was executed."""

	def __init__(self, obs: dict, candidates: list[dict], cand_images: list, flags: PromptFlags):
		self._current_screenshot = obs.get("screenshot")
		self.candidates = candidates
		self.cand_images = cand_images
		self.flags = flags

	@property
	def prompt(self) -> HumanMessage:
		msg = HumanMessage("")
		msg.add_text("# Current state:")
		if self._current_screenshot is not None:
			img_url = image_to_jpg_base64_url(self._current_screenshot)
			msg.add_image(img_url, detail=self.flags.obs.openai_vision_detail)
		msg.add_text(
			"\n# Candidate Actions\n"
			"Below are candidate actions with the resulting browser state after each was executed.\n"
		)
		for i, (cand, screenshot) in enumerate(zip(self.candidates, self.cand_images), 1):
			msg.add_text(
				f"\n## Candidate {i}\n"
				f"Action: `{cand['action']}`\n"
				f"Description: {cand['action_text']}\n"
				f"Resulting state:"
			)
			img_url = image_to_jpg_base64_url(screenshot)
			msg.add_image(img_url, detail=self.flags.obs.openai_vision_detail)
		msg.add_text(
			"\n# Your Task\n"
			"For each candidate, describe in 1-2 sentences what visually changed in the browser "
			"after the action (elements that appeared, disappeared, expanded, or changed). "
			"Focus on observable differences from the current state shown above.\n\n"
			"Respond using this exact format:\n"
			"<effects>\n"
			"<effect_1>Description of what changed after candidate 1.</effect_1>\n"
			"<effect_2>Description of what changed after candidate 2.</effect_2>\n"
			"...\n"
			"</effects>"
		)
		return msg

	def parse_effects(self, text: str) -> list[str]:
		effects = []
		for i in range(1, len(self.candidates) + 1):
			m = re.search(rf"<effect_{i}>(.*?)</effect_{i}>", text, re.DOTALL)
			effects.append(m.group(1).strip() if m else "")
		return effects


class CandidateAwareHistory(dp.PromptElement):
	"""History showing all candidates at each step, with effects and chosen action."""

	def __init__(self, candidate_history: list[dict], thoughts: list[str]):
		super().__init__()
		lines = []
		for step_data, thought in zip(candidate_history, thoughts):
			step = step_data.get("step", "?")
			selected_idx = step_data.get("selected_idx", -1)
			candidates = step_data.get("candidates", [])
			effects = step_data.get("effects", [])
			lines.append(f"\n## Step {step}")
			lines.append("### Candidates explored:")
			for j, cand in enumerate(candidates):
				action = cand.get("action", "")
				action_text = cand.get("action_text", "")
				effect = effects[j] if j < len(effects) else ""
				line = f'C{j + 1}: {action} — "{action_text}"'
				if effect:
					line += f"\n       Effect: {effect}"
				lines.append(line)
			if 0 <= selected_idx < len(candidates):
				chosen = candidates[selected_idx]
				lines.append(f"### Selected: C{selected_idx + 1}")
				lines.append(f'{chosen.get("action", "")} — "{chosen.get("action_text", "")}"')
			lines.append("### Thoughts:")
			lines.append(thought)
		self._prompt = "\n".join(lines) + "\n"


class OracleSelectionPrompt:
	"""Phase 2 (oracle): show K real screenshots from environment execution, ask agent to pick."""

	def __init__(
		self,
		obs: dict,
		actions: list[str],
		thoughts: list[str],
		candidates: list[tuple[str, str, object]],  # [(action_text, action_code, screenshot), ...]
		flags: PromptFlags,
		allow_resample: bool = True,
		effects: list[str] = (),
		include_effects: bool = True,
		include_images: bool = True,
	):
		self.instructions = make_instructions(obs, flags.enable_chat, flags.extra_instructions)
		self.obs = dp.Observation(obs, flags.obs)
		self.history = History(actions, thoughts)
		self.candidates = candidates
		self.flags = flags
		self.allow_resample = allow_resample
		self.effects = list(effects)
		self.include_effects = include_effects
		self.include_images = include_images

	@property
	def prompt(self) -> HumanMessage:
		msg = HumanMessage(self.instructions.prompt)
		msg.add_text(f"{self.obs.prompt}{self.history.prompt}")
		msg = self.obs.add_screenshot(msg)

		header = (
			"\n# Oracle Candidate Actions\n"
			"The following candidate actions were each executed in the real environment. "
		)
		if self.include_images:
			header += "You are shown the resulting screenshot for each.\n"
		else:
			header += "Use the effect descriptions to evaluate each candidate.\n"
		msg.add_text(header)

		for i, (action_text, action_code, screenshot) in enumerate(self.candidates, 1):
			effect = self.effects[i - 1] if i - 1 < len(self.effects) else ""
			msg.add_text(
				f"\n## Candidate {i}\n"
				f"Action code: `{action_code}`\n"
				f"Description: {action_text}\n"
			)
			if self.include_effects and effect:
				msg.add_text(f"Effect: {effect}\n")
			if self.include_images and screenshot is not None:
				msg.add_text("Resulting state:")
				img_url = image_to_jpg_base64_url(screenshot)
				msg.add_image(img_url, detail=self.flags.obs.openai_vision_detail)

		n = len(self.candidates)
		if self.allow_resample:
			msg.add_text(
				f"\n# Your Task\n"
				f"Examine each candidate's resulting state carefully. "
				f"Select the single best candidate (1-{n}) if any of them make meaningful progress toward the goal.\n\n"
				f"However, if **none** of the candidates look productive, you may request a re-sample instead. "
				f"Respond in one of two formats:\n\n"
				f"**To select a candidate:**\n"
				f"<think>\nYour reasoning about which candidate best advances the goal.\n</think>\n"
				f"<selection>N (between 1 and {n})</selection>\n\n"
				f"**To request re-sampling:**\n"
				f"<think>\nExplain why none of the candidates are acceptable. "
				f"Identify specific issues: wrong BIDs, lack of diversity, no progress, etc. "
				f"Suggest what the next round of candidates should try differently.\n</think>\n"
				f"<resample>true</resample>"
			)
		else:
			msg.add_text(
				f"\n# Your Task\n"
				f"Select the single best candidate (1-{n}) based on which resulting "
				f"state best advances the goal. You must generate the thought in the <think> tag before "
				f"selecting the candidate using the <selection> tag.\n\n"
				f"<think>\nYour reasoning about which candidate best advances the goal.\n</think>\n"
				f"<selection>N (between 1 and {n})</selection>"
			)
		return msg

	def parse_answer(self, text: str) -> tuple[int, str]:
		"""Returns (0-indexed selection, reasoning). Raises ResampleRequested if resample chosen."""
		think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
		reasoning = think_m.group(1).strip() if think_m else ""

		resample_m = re.search(r"<resample>\s*(true)\s*</resample>", text, re.IGNORECASE)
		if resample_m and self.allow_resample:
			raise ResampleRequested(reasoning)

		sel_m = re.search(r"<selection>\s*(\d+)\s*</selection>", text, re.DOTALL)
		if not sel_m:
			raise ValueError(f"No <selection> tag found in response: {text[:200]}")
		idx = int(sel_m.group(1)) - 1
		if idx < 0 or idx >= len(self.candidates):
			raise ValueError(f"Selection {idx + 1} out of range (1\u2013{len(self.candidates)})")
		return idx, reasoning


class ResampleRequested(Exception):
	"""Raised by OracleSelectionPrompt.parse_answer when the agent requests re-sampling."""

	def __init__(self, reasoning: str):
		self.reasoning = reasoning
		super().__init__(f"Resample requested: {reasoning[:200]}")