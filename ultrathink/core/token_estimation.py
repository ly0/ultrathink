"""Token estimation utilities for Ultrathink.

This module provides an extensible system for estimating token counts
based on provider and model. Used by both CLI and SDK.
"""

import math
from typing import Callable, Dict, Optional, Tuple

from ultrathink.core.config import ProviderType, get_model_profile


# =============================================================================
# Token Estimation Handlers
# =============================================================================
#
# This is an extensible system for estimating token counts based on provider/model.
# To add support for a new provider or model:
#   1. Define an estimation function: def _estimate_tokens_xxx(text: str) -> int
#   2. Register it in TOKEN_ESTIMATORS dict below
#
# Lookup priority:
#   1. (provider, model) - exact match for specific model
#   2. (provider, None) - fallback for all models of a provider
#   3. Default handler (~4 chars per token)
# =============================================================================

# Type alias for token estimation handler
TokenEstimator = Callable[[str], int]


def _estimate_tokens_default(text: str) -> int:
    """Default token estimation: ~4 characters per token.

    This is a rough approximation suitable for most LLMs.
    """
    return len(text) // 4


def _estimate_tokens_deepseek(text: str) -> int:
    """Estimate token count for DeepSeek models.

    DeepSeek token estimation:
    - 1 English character ≈ 0.3 token
    - 1 Chinese character ≈ 0.6 token

    Returns:
        Token count (rounded up)
    """
    english_chars = 0
    chinese_chars = 0

    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            # CJK Unified Ideographs (Chinese characters)
            chinese_chars += 1
        else:
            english_chars += 1

    tokens = english_chars * 0.3 + chinese_chars * 0.6
    return math.ceil(tokens)


# Token estimator registry
# Key: (provider, model) or (provider, None) for provider-wide handler
# Value: TokenEstimator function
#
# Examples:
#   (ProviderType.DEEPSEEK, None): All DeepSeek models
#   (ProviderType.OPENAI, "gpt-4"): Specific model
#   (ProviderType.ANTHROPIC, None): All Anthropic models
TOKEN_ESTIMATORS: Dict[Tuple[ProviderType, Optional[str]], TokenEstimator] = {
    # DeepSeek: all models use the same estimation
    (ProviderType.DEEPSEEK, None): _estimate_tokens_deepseek,

    # Add more handlers here as needed:
    # (ProviderType.OPENAI, None): _estimate_tokens_openai,
    # (ProviderType.ANTHROPIC, None): _estimate_tokens_anthropic,
    # (ProviderType.OPENAI, "gpt-4"): _estimate_tokens_gpt4,  # Model-specific
}


def get_token_estimator(
    provider: Optional[ProviderType] = None,
    model: Optional[str] = None,
) -> TokenEstimator:
    """Get the appropriate token estimator for a provider/model.

    Lookup priority:
        1. (provider, model) - exact match
        2. (provider, None) - provider-wide handler
        3. Default handler

    Args:
        provider: The provider type
        model: The model name (optional)

    Returns:
        Token estimation function
    """
    # Try exact match first
    if provider is not None and model is not None:
        estimator = TOKEN_ESTIMATORS.get((provider, model))
        if estimator is not None:
            return estimator

    # Try provider-wide handler
    if provider is not None:
        estimator = TOKEN_ESTIMATORS.get((provider, None))
        if estimator is not None:
            return estimator

    # Fall back to default
    return _estimate_tokens_default


def estimate_tokens(
    text: str,
    provider: Optional[ProviderType] = None,
    model: Optional[str] = None,
) -> int:
    """Estimate token count for text based on provider/model.

    Args:
        text: The text to estimate tokens for
        provider: The provider type (uses main profile if None)
        model: The model name (uses main profile if None)

    Returns:
        Estimated token count
    """
    # Get provider/model from main profile if not specified
    if provider is None or model is None:
        profile = get_model_profile("main")
        if profile:
            if provider is None:
                provider = profile.provider
            if model is None:
                model = profile.model

    estimator = get_token_estimator(provider, model)
    return estimator(text)


__all__ = [
    "TokenEstimator",
    "TOKEN_ESTIMATORS",
    "estimate_tokens",
    "get_token_estimator",
]
