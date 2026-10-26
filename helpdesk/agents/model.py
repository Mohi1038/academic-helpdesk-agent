import os

import litellm
from google.adk.models.lite_llm import LiteLlm

from helpdesk import config

# LiteLLM reads the Ollama endpoint from the environment for some of its calls.
os.environ.setdefault("OLLAMA_API_BASE", config.OLLAMA_API_BASE)
litellm.suppress_debug_info = True  # its "Provider List" banners for local models are noise


def make_model(name: str | None = None) -> LiteLlm:
    """An Ollama-hosted chat model for ADK.

    `ollama_chat/` (not `ollama/`) is required: the plain provider loops on tool calls.
    Hybrid reasoning models (qwen3) get thinking switched off - LiteLLM maps
    reasoning_effort="none" to Ollama's `think: false` - which keeps turns fast and short.
    """
    name = name or config.CHAT_MODEL
    kwargs = {"api_base": config.OLLAMA_API_BASE, "num_ctx": config.NUM_CTX}
    if config.TEMPERATURE is not None:
        kwargs["temperature"] = config.TEMPERATURE
    if name.startswith("qwen3"):
        kwargs["reasoning_effort"] = "none"
    return LiteLlm(model=f"ollama_chat/{name}", **kwargs)
