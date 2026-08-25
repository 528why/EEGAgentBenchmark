"""OpenAI-compatible agent adapter registration."""

from eeg_agent_bench.agents.adapters.openai_compatible import OpenAICompatibleAdapter
from eeg_agent_bench.agents.base import AgentAdapterFactory

AgentAdapterFactory.register("openai_compatible", OpenAICompatibleAdapter)
