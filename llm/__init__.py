from .base import LLMClient
from config import LLM_PROVIDER, API_KEY, BASE_URL, MODEL, LLM_TIMEOUT


def get_client(model_override: str = None) -> LLMClient:
    """Factory: create LLM client based on LLM_PROVIDER config.

    Args:
        model_override: Use a different model than the default (e.g. for proof_tool).
    """
    model = model_override or MODEL

    if LLM_PROVIDER == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(api_key=API_KEY, base_url=BASE_URL, model=model, timeout=LLM_TIMEOUT)
    else:
        from .openai_client import OpenAIClient
        return OpenAIClient(api_key=API_KEY, base_url=BASE_URL, model=model, timeout=LLM_TIMEOUT)
