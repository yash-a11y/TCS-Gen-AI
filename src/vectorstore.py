from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np
from src.config import EMBEDDING_MODEL, FAISS_DIR, ensure_data_dirs
from src.embedding import get_embedding_pipeline

_store: FaissVectorStore | None = None


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.emb_pipe = get_embedding_pipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def build_from_documents(self, documents: List[Any]):
        print(f"Building vector store from {len(documents)} raw documents...")
        chunks = self.emb_pipe.chunk_documents(documents)
        embeddings = self.emb_pipe.embed_chunks(chunks)
        metadatas = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            metadatas.append(
                {
                    "text": chunk.page_content,
                    "source": Path(str(source)).name if source else None,
                    "page": chunk.metadata.get("page"),
                }
            )
        self.index = None
        self.metadata = []
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()
        print(f"Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        if embeddings.size == 0:
            return
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not os.path.exists(faiss_path):
            print(f"No Faiss index yet at {faiss_path}")
            return
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None or self.index.ntotal == 0:
            return []
        vector = np.asarray(query_embedding).astype("float32")
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(vector, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"Querying vector store for: '{query_text}'")
        query_emb = self.emb_pipe.embed_query(query_text)
        return self.search(query_emb, top_k=top_k)


def get_vector_store() -> FaissVectorStore:
    global _store
    ensure_data_dirs()
    if _store is None:
        _store = FaissVectorStore(persist_dir=str(FAISS_DIR), embedding_model=EMBEDDING_MODEL)
        _store.load()
    return _store
