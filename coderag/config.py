from __future__ import annotations

import os

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


AI_BACKEND: str = _get_env("AI_BACKEND", default="ollama")

OPENAI_API_KEY: str = _get_env("API_KEY", default="ollama")
OPENAI_BASE_URL: str = _get_env("BASE_URL",default="http://localhost:11434/v1/",)
CHAT_MODEL: str = _get_env("CHAT_MODEL", default="gemma3:latest")
EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL",default="nomic-embed-text",)

WATCHED_DIR: Path = Path(_get_env("WATCHED_DIR", default=".")).expanduser().resolve()
FAISS_INDEX_FILE: Path = Path(_get_env("FAISS_INDEX_FILE", default="./coderag_index.faiss")).expanduser().resolve()

EMBEDDING_DIM: int = int(_get_env("EMBEDDING_DIM", default="768"))
METADATA_FILE: Path = FAISS_INDEX_FILE.with_name("metadata.npy")


#This creates a set of folder names to skip during indexing.
IGNORE_DIRS: set[str] = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "tests",
}

ALLOWED_EXTENSIONS: set[str] = {
    ".py",
}