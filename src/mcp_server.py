"""MCP server: SQL customer tools + policy search (no Groq)."""

from urllib.parse import urlparse

from fastmcp import FastMCP

from src.config import MCP_SERVER_URL
from src.data_loader import ingest_pdf as ingest_pdf_file
from src.db import get_connection, get_customer_profile as db_profile
from src.db import get_customer_tickets as db_tickets
from src.db import rows_to_dicts
from src.search import search_policies as rag_search_policies

mcp = FastMCP(
    "acme-support",
    instructions=(
        "Tools for Acme customer support: look up customers/tickets in SQLite "
        "and search company policy PDFs."
    ),
)


@mcp.tool
def get_customer_profile(name: str) -> dict:
    """Look up one customer by name (partial match)."""
    profile = db_profile(name)
    if profile is None:
        return {"found": False, "query": name}
    return {"found": True, **profile}


@mcp.tool
def get_customer_tickets(name: str) -> dict:
    """Return support tickets for a customer by name."""
    profile = db_profile(name)
    if profile is None:
        return {"found": False, "query": name, "tickets": []}
    tickets = db_tickets(name)
    return {
        "found": True,
        "customer": profile["name"],
        "customer_id": profile["id"],
        "tickets": tickets,
    }


@mcp.tool
def search_policies(query: str, k: int = 4) -> dict:
    """Search uploaded/seeded policy PDFs. Returns relevant text chunks."""
    hits = rag_search_policies(query, k=k)
    for hit in hits:
        if hit.get("distance") is not None:
            hit["distance"] = float(hit["distance"])
    return {"query": query, "hits": hits}


@mcp.tool
def ingest_pdf(pdf_path: str) -> dict:
    """Add a policy PDF into the FAISS store (re-indexes that folder)."""
    return ingest_pdf_file(pdf_path)


@mcp.tool
def run_readonly_sql(sql: str) -> dict:
    """Run a read-only SELECT on the customer SQLite database."""
    stripped = sql.strip().rstrip(";")
    if not stripped.lower().startswith("select"):
        return {"ok": False, "error": "Only SELECT statements are allowed."}
    conn = get_connection()
    try:
        rows = conn.execute(stripped).fetchall()
        return {"ok": True, "rows": rows_to_dicts(rows)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        conn.close()


def main() -> None:
    parsed = urlparse(MCP_SERVER_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    path = parsed.path or "/mcp"
    print(f"Starting MCP HTTP server at http://{host}:{port}{path}")
    mcp.run(transport="http", host=host, port=port, path=path)


if __name__ == "__main__":
    import sys

    if "--http" in sys.argv:
        main()
    else:
        mcp.run(transport="stdio")
