import logging
from typing import List, Optional

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from coderag.ai_client import get_ai_client
from coderag.config import EMBEDDING_DIM, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def _chunk_text(text: str, max_chars: int = 4000, overlap: int = 400) -> List[str]:
    """Chunk text with small overlap to preserve boundary context."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    step = max_chars - overlap
    return [text[i : i + max_chars] for i in range(0, len(text), step)]
    """  [0:4000], [3600:7600], [7200:11200], ... """





#decorator for the func below
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),  # exponential backoff: 0.5, 1, 2, 4, 8 seconds
    reraise=True,   # re-raise the last exception after all retries are exhausted
)
def _embed_batch(inputs: List[str]) -> np.ndarray:
    """Call embeddings API with retry/backoff. Returns shape (n, d)."""
    client, provider = get_ai_client() #provider not used 
    if client is None:
        raise RuntimeError("AI client not initialized")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=inputs,
        timeout=30,
    )
    return np.array([d.embedding for d in response.data], dtype=np.float32)
    #eg.response.data = [{"index": 0, "embedding": [0.12, 0.45, -0.09]},{"index": 1, "embedding": [0.33, 0.11, 0.77]},]
    #float 32 for fiass





def generate_embeddings(text: str) -> Optional[np.ndarray]:
    """Generate one averaged embedding vector shaped (1, d)."""
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding generation")
        return None

    try:
        chunks = _chunk_text(text)
        if not chunks:
            return None

        vecs = _embed_batch(chunks)

        #axis 0 means average accross rows (col by col)
        # reshape does this (768,) -> (1, 768).   syntax(row, col). -1 means figure it out
        avg = np.mean(vecs, axis=0, dtype=np.float32).reshape(1, -1)

        if avg.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Configured EMBEDDING_DIM={EMBEDDING_DIM}, "
                f"but model '{EMBEDDING_MODEL}' returned {avg.shape[1]}"
            )

        return avg.astype(np.float32)

    except Exception as e:
        logger.error("Failed to generate embeddings: %s", e)
        return None