"""Retrieve policy chunks from FAISS and summarize them with Groq."""

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL, POLICY_DIR
from src.llm import get_llm
from src.vectorstore import get_vector_store


class RAGSearch:
    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
    ):
        self.llm_model = llm_model or GROQ_MODEL
        self.vectorstore = get_vector_store()

        if self.vectorstore.index is None:
            from src.data_loader import load_all_documents

            docs = load_all_documents(str(POLICY_DIR))
            self.vectorstore.build_from_documents(docs)

        self.llm = None

    def _get_llm(self) -> ChatGroq:
        if self.llm is None:
            if self.llm_model == GROQ_MODEL:
                self.llm = get_llm()
            else:
                if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
                    raise RuntimeError(
                        "GROQ_API_KEY is missing. Copy .env.example to .env and paste your key."
                    )
                self.llm = ChatGroq(model=self.llm_model, api_key=GROQ_API_KEY, temperature=0)
            print(f"Groq LLM initialized: {self.llm_model}")
        return self.llm

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = (
            f"Summarize the following context for the query: '{query}'\n\n"
            f"Context:\n{context}\n\n"
            "Summary:"
        )
        response = self._get_llm().invoke(prompt)
        return response.content


_rag: RAGSearch | None = None


def get_rag() -> RAGSearch:
    global _rag
    if _rag is None:
        _rag = RAGSearch()
    return _rag


def search_policies(query: str, k: int = 4) -> list[dict]:
    hits = get_rag().vectorstore.query(query, top_k=k)
    results = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        results.append(
            {
                "text": meta.get("text"),
                "source": meta.get("source"),
                "page": meta.get("page"),
                "distance": hit.get("distance"),
            }
        )
    return results


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is the current refund policy?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
