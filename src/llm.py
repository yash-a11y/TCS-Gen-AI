"""Groq chat model used by the LangGraph agents."""

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL


def get_llm(*, temperature: float = 0) -> ChatGroq:
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and paste your key."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )
