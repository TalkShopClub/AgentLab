"""Shared setup for diagnostic experiment scripts.

Mirrors oracle_loop.py lines 411-441 so each experiment doesn't duplicate the boilerplate.
"""

import copy

from bgym import HighLevelActionSetArgs

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent import AGENT_GPT5
from agentlab.agents.wm_visual_agent.agent_configs import DEFAULT_OBS_FLAGS, DEFAULT_PROMPT_FLAGS
from agentlab.experiments.loop import EnvArgs
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from oracle_wm._bid_utils import get_valid_snow_instance


def setup_experiment(task, seed, model, headless, agent_mode):
    """Build flags, action_set, obs_preprocessor, chat_llm, env_args, instance.

    Returns dict with keys: flags, oracle_sel_flags, action_set, obs_preprocessor,
    chat_llm, env_args, instance.
    """
    if agent_mode == "text":
        flags = copy.deepcopy(AGENT_GPT5.flags)
        flags.action.long_description = True
        action_set = HighLevelActionSetArgs(subsets=["workarena++"]).make_action_set()
    else:
        flags = copy.deepcopy(DEFAULT_PROMPT_FLAGS)
        action_set = HighLevelActionSetArgs(subsets=["coord", "workarena++"]).make_action_set()

    if task.endswith("-l3"):
        flags.use_memory = True

    oracle_sel_flags = copy.deepcopy(flags)
    oracle_sel_flags.obs.use_screenshot = True
    oracle_sel_flags.obs.use_som = False

    preproc_obs_flags = copy.deepcopy(DEFAULT_OBS_FLAGS)
    preproc_obs_flags.use_ax_tree = True
    obs_preprocessor = dp.make_obs_preprocessor(preproc_obs_flags)

    chat_llm = CHAT_MODEL_ARGS_DICT[model].make_model()
    env_args = EnvArgs(task_name=task, task_seed=seed, headless=headless)
    instance = get_valid_snow_instance()

    return {
        "flags": flags,
        "oracle_sel_flags": oracle_sel_flags,
        "action_set": action_set,
        "obs_preprocessor": obs_preprocessor,
        "chat_llm": chat_llm,
        "env_args": env_args,
        "instance": instance,
    }
