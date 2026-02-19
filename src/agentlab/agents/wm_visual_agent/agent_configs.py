import bgym
from bgym import HighLevelActionSetArgs

import agentlab.agents.dynamic_prompting as dp
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .wm_visual_agent import WMVisualAgentArgs
from agentlab.agents.visual_agent.visual_agent_prompts import PromptFlags

DEFAULT_OBS_FLAGS = dp.ObsFlags(
    use_tabs=True,
    use_error_logs=True,
    use_past_error_logs=False,
    use_screenshot=True,
    use_som=True,
    use_history=True,
    use_action_history=True,
    use_think_history=True,
    openai_vision_detail="high",
)

DEFAULT_ACTION_FLAGS = dp.ActionFlags(
    action_set=HighLevelActionSetArgs(subsets=["coord", "bid"]),
    long_description=True,
    individual_examples=False,
)

DEFAULT_PROMPT_FLAGS = PromptFlags(
    obs=DEFAULT_OBS_FLAGS,
    action=DEFAULT_ACTION_FLAGS,
    use_thinking=True,
    use_concrete_example=False,
    use_abstract_example=True,
    enable_chat=False,
    extra_instructions=None,
)

WM_VISUAL_AGENT_GPT5_IMAGE = WMVisualAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-5-2025-08-07"],
    flags=DEFAULT_PROMPT_FLAGS,
    wm_server_url="https://z66y0a4p8qruii-8000.proxy.runpod.net/",
    wm_mode="image",
)

WM_VISUAL_AGENT_GPT5_TEXT = WMVisualAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-5-2025-08-07"],
    flags=DEFAULT_PROMPT_FLAGS,
    wm_server_url="https://z66y0a4p8qruii-8000.proxy.runpod.net/",
    wm_mode="text",
)