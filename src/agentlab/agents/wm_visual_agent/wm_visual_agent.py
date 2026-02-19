"""World-model-augmented visual agent.

Two-phase action selection:
  Phase 1: LLM proposes 5 candidate actions (code + text description + rationale)
  Phase 2: Emu3.5 predicts next state for each candidate (using action_text),
           predictions are shown to the LLM which picks the best action.
"""

import logging
from dataclasses import asdict, dataclass

import bgym
from bgym import Benchmark
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.world_model_client import WorldModelClient
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from .wm_prompts import CandidateGenerationPrompt, InformedSelectionPrompt
from agentlab.agents.visual_agent.visual_agent_prompts import PromptFlags

logger = logging.getLogger(__name__)


@dataclass
class WMVisualAgentArgs(AgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: PromptFlags = None
    max_retry: int = 4
    wm_server_url: str = "http://localhost:8000"
    wm_mode: str = "image"  # "image" or "text"
    wm_timeout: int = 600

    def __post_init__(self):
        try:
            self.agent_name = f"WMVisualAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: Benchmark, demo_mode):
        self.flags.obs.use_tabs = benchmark.is_multi_tab

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        return WMVisualAgent(
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
            wm_server_url=self.wm_server_url,
            wm_mode=self.wm_mode,
            wm_timeout=self.wm_timeout,
        )


class WMVisualAgent(Agent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: PromptFlags,
        max_retry: int = 4,
        wm_server_url: str = "http://localhost:8000",
        wm_mode: str = "image",
        wm_timeout: int = 600,
    ):
        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry
        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)
        self.wm_client = WorldModelClient(
            server_url=wm_server_url, mode=wm_mode, timeout=wm_timeout
        )
        self.wm_mode = wm_mode
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):
        # --- Phase 1: Generate 5 candidate actions ---
        candidates = self._phase1_candidates(obs)

        if not candidates:
            raise RuntimeError("Phase 1 candidate generation returned no candidates")

        # --- World Model: get predictions using action_text ---
        screenshot = obs.get("screenshot_som") if self.flags.obs.use_som else obs.get("screenshot")
        if screenshot is None:
            screenshot = obs.get("screenshot")

        action_texts = [c["action_text"] for c in candidates]
        predictions = self.wm_client.predict_batch(screenshot, action_texts)

        # --- Phase 2: Informed selection ---
        action, ans_dict, phase2_messages = self._phase2_selection(obs, candidates, predictions)

        self.actions.append(action)
        self.thoughts.append(ans_dict.get("think", None))

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict.get("n_retry", 0)
        stats["busted_retry"] = ans_dict.get("busted_retry", 0)
        stats["n_candidates"] = len(candidates)
        stats["wm_mode"] = self.wm_mode

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=phase2_messages,
            stats=stats,
            extra_info={
                "chat_model_args": asdict(self.chat_model_args),
                "wm_candidates": candidates,
                "wm_predictions": predictions,
                "wm_mode": self.wm_mode,
            },
        )
        return action, agent_info

    def _phase1_candidates(self, obs) -> list[dict]:
        """Ask the LLM for 5 candidate actions."""
        gen_prompt = CandidateGenerationPrompt(
            action_set=self.action_set,
            obs=obs,
            actions=self.actions,
            thoughts=self.thoughts,
            flags=self.flags,
            model_name=self.chat_model_args.model_name,
        )
        system_prompt = SystemMessage(
            "You are an agent trying to solve a web task. "
            "Propose your top 5 candidate actions for the current state."
        )
        chat_messages = Discussion([system_prompt, gen_prompt.prompt])

        response = self.chat_llm(chat_messages.messages)
        text = response.content if hasattr(response, "content") else str(response)
        if hasattr(response, "choices"):
            text = response.choices[0].message.content
        candidates = gen_prompt.parse_candidates(text)
        return candidates

    def _phase2_selection(self, obs, candidates, predictions):
        """Present predictions to the LLM and pick the best action."""
        sel_prompt = InformedSelectionPrompt(
            action_set=self.action_set,
            obs=obs,
            candidates=candidates,
            predictions=predictions,
            flags=self.flags,
            mode=self.wm_mode,
            model_name=self.chat_model_args.model_name,
        )
        system_prompt = SystemMessage(
            "You are an agent trying to solve a web task. "
            "A world model has predicted the next state for each candidate action. "
            "Select the best action based on the predictions."
        )
        chat_messages = Discussion([system_prompt, sel_prompt.prompt])

        try:
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=sel_prompt.parse_answer,
            )
            ans_dict["busted_retry"] = 0
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError:
            ans_dict = dict(action=None, n_retry=self.max_retry + 1, busted_retry=1)

        return ans_dict.get("action"), ans_dict, chat_messages

    def reset(self, seed=None):
        self.seed = seed
        self.thoughts = []
        self.actions = []
