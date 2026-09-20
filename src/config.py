"""Shared paths and environment settings."""

from __future__ import annotations


import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent


load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
POLICY_DIR = DATA_DIR / "policies"
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss_store"
DB_PATH = DATA_DIR / "support.db"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, POLICY_DIR, UPLOAD_DIR, FAISS_DIR):
        path.mkdir(parents=True, exist_ok=True)
