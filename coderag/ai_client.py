import logging
from typing import Optional, Tuple

from openai import OpenAI

from coderag.config import AI_BACKEND, OPENAI_API_KEY, OPENAI_BASE_URL

logger = logging.getLogger(__name__)


def get_ai_client() -> Tuple[Optional[OpenAI], str]:
    """
    Create a generic OpenAI-compatible client.

    Backends:
    - "local"  → uses BASE_URL (local)
    - "remote" → uses API key (hosted service)
    """

    try:
        backend = AI_BACKEND.strip().lower()

        # --- Local backend (uses base URL) ---
        if backend == "local":
            base_url = OPENAI_BASE_URL or "http://localhost:11434/v1/"
            api_key = OPENAI_API_KEY or "local-key"

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

            provider = f"Local backend ({base_url})"
            logger.info("Using local backend")
            return client, provider

        # --- Remote backend (uses API key) ---
        elif backend == "remote":
            if not OPENAI_API_KEY:
                logger.error("Missing API key for remote backend")
                return None, "Unavailable"

            client = OpenAI(api_key=OPENAI_API_KEY)
            provider = "Remote backend"

            logger.info("Using remote backend")
            return client, provider

        # --- Unknown backend ---
        else:
            logger.error(f"Unsupported AI_BACKEND: {AI_BACKEND}")
            return None, "Unavailable"

    except Exception as e:
        logger.error("Failed to initialize AI client: %s", e)
        return None, "Unavailable"