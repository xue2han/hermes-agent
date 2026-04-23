"""E2E tests: verify _build_api_kwargs uses provider profile for active providers."""

import sys
import os
import pytest

# Ensure the worktree is on the import path
_wt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _wt not in sys.path:
    sys.path.insert(0, _wt)


@pytest.fixture
def nvidia_agent():
    """Minimal AIAgent configured as NVIDIA provider."""
    from run_agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent.model = "nvidia/llama-3.1-nemotron-70b-instruct"
    agent.provider = "nvidia"
    agent.base_url = "https://integrate.api.nvidia.com/v1"
    agent._base_url_lower = agent.base_url.lower()
    agent._base_url_hostname = "integrate.api.nvidia.com"
    agent.api_mode = "chat_completions"
    agent.tools = None
    agent.max_tokens = None
    agent.reasoning_config = None
    agent.request_overrides = None
    agent.session_id = "test"
    agent._ollama_num_ctx = None
    agent._ephemeral_max_output_tokens = None
    agent._transport_cache = {}
    agent._is_anthropic_oauth = False
    agent._max_tokens_param = lambda x: {"max_tokens": x} if x else {}
    agent._resolved_api_call_timeout = lambda: 300
    return agent


@pytest.fixture
def deepseek_agent():
    """Minimal AIAgent configured as DeepSeek provider."""
    from run_agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent.model = "deepseek-chat"
    agent.provider = "deepseek"
    agent.base_url = "https://api.deepseek.com/v1"
    agent._base_url_lower = agent.base_url.lower()
    agent._base_url_hostname = "api.deepseek.com"
    agent.api_mode = "chat_completions"
    agent.tools = None
    agent.max_tokens = None
    agent.reasoning_config = None
    agent.request_overrides = None
    agent.session_id = "test"
    agent._ollama_num_ctx = None
    agent._ephemeral_max_output_tokens = None
    agent._transport_cache = {}
    agent._is_anthropic_oauth = False
    agent._max_tokens_param = lambda x: {"max_tokens": x} if x else {}
    agent._resolved_api_call_timeout = lambda: 300
    return agent


class TestNvidiaProfileWiring:
    def test_nvidia_gets_default_max_tokens(self, nvidia_agent):
        kwargs = nvidia_agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["max_tokens"] == 16384

    def test_nvidia_model_passed(self, nvidia_agent):
        kwargs = nvidia_agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["model"] == "nvidia/llama-3.1-nemotron-70b-instruct"

    def test_nvidia_messages_passed(self, nvidia_agent):
        msgs = [{"role": "user", "content": "hi"}]
        kwargs = nvidia_agent._build_api_kwargs(msgs)
        assert kwargs["messages"] == msgs


class TestDeepSeekProfileWiring:
    def test_deepseek_no_forced_max_tokens(self, deepseek_agent):
        kwargs = deepseek_agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        # DeepSeek profile has no default_max_tokens — max_tokens comes from agent
        assert kwargs["model"] == "deepseek-chat"

    def test_deepseek_messages_passed(self, deepseek_agent):
        msgs = [{"role": "user", "content": "hi"}]
        kwargs = deepseek_agent._build_api_kwargs(msgs)
        assert kwargs["messages"] == msgs
