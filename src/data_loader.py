"""Load PDFs with LangChain PyPDFLoader, then ingest into FAISS."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from src.vectorstore import get_vector_store


def load_all_documents(data_dir: str):
    """Load every PDF under data_dir into LangChain Documents."""
    data_path = Path(data_dir).resolve()
    documents = []

    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        try:
            loaded = PyPDFLoader(str(pdf_file)).load()
            print(f"Loaded {len(loaded)} PDF docs from {pdf_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Failed to load PDF {pdf_file}: {e}")

    print(f"Total loaded documents: {len(documents)}")
    return documents


def ingest_directory(directory):
    docs = load_all_documents(str(directory))
    store = get_vector_store()
    store.build_from_documents(docs)

    counts: dict[str, int] = {}
    for meta in store.metadata:
        source = meta.get("source") or "unknown"
        counts[source] = counts.get(source, 0) + 1
    return [{"source": source, "chunks": n} for source, n in sorted(counts.items())]


def ingest_pdf(pdf_path, source_name=None):
    """Re-index every PDF in the same folder so one file does not wipe the store."""
    path = Path(pdf_path)
    source = source_name or path.name
    pages = PyPDFLoader(str(path)).load()
    if not pages:
        return {"source": source, "chunks": 0}
    ingest_directory(path.parent)
    return {"source": source, "chunks": len(get_vector_store().metadata)}


if __name__ == "__main__":
    from src.config import POLICY_DIR

    docs = load_all_documents(str(POLICY_DIR))
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
