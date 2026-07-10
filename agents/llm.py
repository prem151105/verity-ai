"""
Shared LLM client — singleton so all agents use the same Gemini model instance.
Uses google-genai (google.genai) — the current supported package.
Includes exponential backoff for free-tier rate limits (5 RPM).
"""

import time
import logging
from config import settings

logger = logging.getLogger(__name__)

_client = None

# Free tier: 15 RPM but in practice multi-agent pipelines hit limits fast.
# Add a minimum delay between calls to stay under quota.
_MIN_CALL_INTERVAL_S = 4.0   # 15 RPM ≈ one call per 4s; safe for multi-agent
_last_call_time = 0.0


def get_llm_client():
    """Return a cached google.genai.Client instance."""
    global _client
    if _client is None:
        try:
            from google import genai
            _client = genai.Client(api_key=settings.gemini_api_key)
            logger.info(f"Initialized Gemini client | model: {settings.gemini_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    return _client


def call_llm(prompt: str, system_instruction: str = "") -> str:
    """
    Text generation call via google.genai with exponential backoff.
    Handles 429 RESOURCE_EXHAUSTED from free-tier rate limits gracefully.

    Args:
        prompt: The user prompt
        system_instruction: Optional system-level instruction (prepended)

    Returns:
        Model response text
    """
    global _last_call_time
    from google.genai import types as genai_types
    from google.genai.errors import ClientError

    client = get_llm_client()

    if system_instruction:
        full_prompt = f"[SYSTEM INSTRUCTION]\n{system_instruction}\n\n[USER]\n{prompt}"
    else:
        full_prompt = prompt

    # Rate-limit guard: enforce minimum interval between calls
    elapsed = time.monotonic() - _last_call_time
    if elapsed < _MIN_CALL_INTERVAL_S:
        sleep_time = _MIN_CALL_INTERVAL_S - elapsed
        logger.debug(f"[LLM] Rate-limit guard: sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)

    max_retries = 5
    backoff = 20.0  # start at 20s (matches the retry delay from API error)

    for attempt in range(max_retries):
        try:
            _last_call_time = time.monotonic()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.95,
                    max_output_tokens=8192,
                ),
            )
            return response.text.strip()

        except ClientError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = backoff * (2 ** attempt)
                logger.warning(
                    f"[LLM] Rate limited (429). Waiting {wait:.0f}s before retry "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            logger.error(f"[LLM] Unexpected error: {e}")
            raise
