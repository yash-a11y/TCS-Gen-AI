from __future__ import annotations

from typing import Any, List

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

_pipeline: EmbeddingPipeline | None = None


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Embeddings shape: {embeddings.shape}")
        return np.asarray(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(self.model.encode([query])).astype("float32")


def get_embedding_pipeline(
    model_name: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> EmbeddingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = EmbeddingPipeline(
            model_name=model_name or EMBEDDING_MODEL,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return _pipeline


if __name__ == "__main__":
    from src.config import POLICY_DIR
    from src.data_loader import load_all_documents

    docs = load_all_documents(str(POLICY_DIR))
    emb_pipe = EmbeddingPipeline(model_name=EMBEDDING_MODEL)
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
