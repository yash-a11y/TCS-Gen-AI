# Support Assistant

Generative AI multi-agent system for a customer-support workflow. A support agent (John) can ask natural-language questions about **structured customer data** (SQLite) and **unstructured policy PDFs** (RAG over FAISS). A LangGraph supervisor routes each question to a SQL specialist, a policy specialist, or both. Tools are exposed through an **MCP server**.

This repository is a local demo: synthetic customers (including **Ema Patel**), sample Acme policy PDFs, and a Streamlit chat UI.

---

## What it does

| You ask | What happens |
| --- | --- |
| "What is the current refund policy?" | RAG agent searches policy PDFs |
| "Give me an overview of customer Ema and her tickets" | SQL agent reads SQLite |
| "Does Ema's refund ticket match the refund policy?" | Hybrid: SQL + RAG, then one answer |
| Upload a new policy PDF in the sidebar | Document is chunked, embedded, and indexed |

---

## Architecture

```text
Streamlit chat
      |
      v
LangGraph supervisor --> sql | rag | hybrid
      |
      v
MCP tools (FastMCP)
      |-- get_customer_profile / get_customer_tickets / run_readonly_sql
      |-- search_policies / ingest_pdf
              |
              |-- SQLite   data/support.db
              |-- FAISS    data/faiss_store
```

By default the graph talks to MCP **in-process** (one command for the UI). You can also run MCP as a separate HTTP process for a two-terminal demo.

---

## Tech stack

| Layer | Choice |
| --- | --- |
| Agents | LangChain, LangGraph |
| LLM | Groq (`openai/gpt-oss-20b` by default) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Structured data | SQLite |
| Unstructured data | FAISS + MiniLM |
| Tool server | FastMCP |
| UI | Streamlit |

---

## Prerequisites

- **Python 3.11+** (developed on 3.13)
- A free **Groq API key**: [https://console.groq.com/keys](https://console.groq.com/keys)
- Disk space for PyTorch / sentence-transformers (first install can take several minutes)

---

## Setup (local)

All commands assume the project root (the folder that contains `requirements.txt`).

### 1. Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

```powershell
copy .env.example .env
```

```bash
cp .env.example .env
```

Edit `.env` and set your key:

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-20b
MCP_SERVER_URL=http://127.0.0.1:8000/mcp
```

`llama-3.3-70b-versatile` is retired on Groq free/developer tiers. If a model returns HTTP 404, list models your key can use and pick one that supports tool calling, for example `openai/gpt-oss-20b` or `openai/gpt-oss-120b`.

### 3. Seed the database, PDFs, and vector index

```powershell
.\.venv\Scripts\python.exe -m src.seed
```

```bash
python -m src.seed
```

This creates:

- `data/support.db` -- 15 customers, 24 tickets (including Ema)
- `data/policies/*.pdf` -- refund, privacy, and SLA samples
- `data/faiss_store/` -- FAISS index + metadata

The first seed download of MiniLM can take a minute.

### 4. Check the environment (optional)

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

You should see `Packages: OK` and `GROQ_API_KEY: set`.

---

## Run locally

### Chat UI (main demo)

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

```bash
python -m streamlit run app/streamlit_app.py
```

Open **http://localhost:8501**.

- Type a question in the chat box, or use the sidebar starters.
- Each answer shows which agent ran (`sql`, `rag`, or `hybrid`).
- Upload a policy PDF in the sidebar to add it to the knowledge base (re-indexes `data/policies`).

The UI does **not** require a separate MCP process. LangGraph calls FastMCP in-process.

### CLI (no browser)

```powershell
.\.venv\Scripts\python.exe -m src.graph "What is the current refund policy?"
.\.venv\Scripts\python.exe -m src.graph "Give me a quick overview of customer Ema and her tickets"
.\.venv\Scripts\python.exe -m src.graph "Does Ema's refund ticket match the current refund policy?"
```

### MCP HTTP server (optional two-process demo)

**Terminal A**

```powershell
.\.venv\Scripts\python.exe -m src.mcp_server --http
```

Server: `http://127.0.0.1:8000/mcp`

**Terminal B**

```powershell
$env:MCP_TRANSPORT="http"
.\.venv\Scripts\python.exe -m src.graph "What is the current refund policy?"
```

```bash
MCP_TRANSPORT=http python -m src.graph "What is the current refund policy?"
```

Without `--http`, `python -m src.mcp_server` runs **stdio** MCP (for clients that spawn the process).

---

## Sample questions

Use these in Streamlit or with `python -m src.graph "..."`:

1. What is the current refund policy?
2. Give me a quick overview of customer Ema and her past support tickets.
3. Does Ema's refund ticket match the current refund policy?
4. What is the Enterprise P1 first-response time?
5. Who is the account manager for Ema Patel?

---

## Project layout

```text
app/streamlit_app.py      Chat UI
src/config.py             Paths and Groq/MCP settings
src/db.py                 SQLite helpers (profile, tickets)
src/seed.py               Synthetic data + sample PDFs
src/data_loader.py        PyPDFLoader + FAISS ingest
src/embedding.py          Chunk + MiniLM embeddings
src/vectorstore.py        FAISS index
src/search.py             Retrieve / summarize policies
src/mcp_server.py         FastMCP tools
src/graph.py              Supervisor + SQL/RAG/hybrid agents
src/llm.py                ChatGroq wrapper
data/policies/            Policy PDFs
data/support.db           Created by seed
data/faiss_store/         Created by seed
.env.example              Template for secrets
```

---

## Quick verification

| Check | Command | Expected |
| --- | --- | --- |
| Ema in SQL | `python -m src.db` | Ema Patel, 5 tickets |
| Load PDFs | `python -m src.data_loader` | 3 (or more) PDFs |
| Seed | `python -m src.seed` | `Seed complete.` |
| Graph | `python -m src.graph "What is the current refund policy?"` | `route: rag` and a policy summary |

---

## Troubleshooting

**`GROQ_API_KEY is missing`**  
Copy `.env.example` to `.env` and paste a key from the Groq console.

**`The model ... does not exist` (404)**  
That model ID is not available on your Groq plan. Set `GROQ_MODEL` in `.env` to `openai/gpt-oss-20b` or another ID from [Groq models](https://console.groq.com/docs/models).

**Unicode error when printing in Windows PowerShell**  
The graph CLI already reconfigures stdout to UTF-8. Streamlit is unaffected. If you print answers in a one-liner, use UTF-8 or avoid the console.

**First pip install is slow**  
`torch` and `sentence-transformers` are large. Later installs reuse the venv cache.

**Streamlit cannot import `src`**  
Always start Streamlit from the **project root**, not from `app/`.

**Empty or stale policy answers after adding a PDF**  
Use the sidebar uploader (it re-indexes the policies folder) or run `python -m src.seed` again.

---

## License / data

Sample customers and tickets are synthetic. Sample PDFs are generated Acme policies for demonstration only. Do not commit a real `.env` file.
