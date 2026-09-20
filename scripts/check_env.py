"""Verify the local environment: venv packages, .env, Groq key."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = [
    "langchain",
    "langchain_groq",
    "langgraph",
    "mcp",
    "fastmcp",
    "faiss",
    "sentence_transformers",
    "streamlit",
    "pypdf",
    "reportlab",
    "dotenv",
]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Activate the venv and run: pip install -r requirements.txt")
        return 1
    print("Packages: OK")

    env_path = ROOT / ".env"
    if not env_path.exists():
        print("No .env file yet. Copy .env.example to .env and add GROQ_API_KEY.")
        return 0

    from src.config import GROQ_API_KEY, GROQ_MODEL, MCP_SERVER_URL

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("GROQ_API_KEY is not set in .env")
        return 1

    print(f"GROQ_MODEL: {GROQ_MODEL}")
    print(f"MCP_SERVER_URL: {MCP_SERVER_URL}")
    print("GROQ_API_KEY: set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
