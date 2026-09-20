"""LangGraph multi-agent: supervisor routes to SQL, RAG, or hybrid via MCP tools."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.config import MCP_SERVER_URL
from src.llm import get_llm

SQL_TOOL_NAMES = {"get_customer_profile", "get_customer_tickets", "run_readonly_sql"}
RAG_TOOL_NAMES = {"search_policies", "ingest_pdf"}


class RouteDecision(BaseModel):
    next: Literal["sql", "rag", "hybrid"] = Field(
        description="sql = customer/ticket database; rag = policy PDFs; hybrid = both"
    )


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str


def _tool_payload(result) -> str:
    data = result.data if getattr(result, "data", None) is not None else result
    if isinstance(data, (dict, list)):
        return json.dumps(data, default=str)
    return str(data)


async def _mcp_tools():
    """Load MCP tools: HTTP if MCP_TRANSPORT=http, else in-process FastMCP."""
    if os.getenv("MCP_TRANSPORT", "inprocess") == "http":
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "acme": {
                    "transport": "http",
                    "url": MCP_SERVER_URL,
                }
            }
        )
        return await client.get_tools()

    from fastmcp import Client

    from src.mcp_server import mcp

    async with Client(mcp) as session:
        specs = await session.list_tools()

    tools = []
    for spec in specs:
        async def _call(*args, _name=spec.name, _schema=spec.inputSchema, **kwargs):
            # LangChain may pass a single dict positional arg
            if args and isinstance(args[0], dict) and not kwargs:
                kwargs = args[0]
            from fastmcp import Client as FastClient
            from src.mcp_server import mcp as server

            async with FastClient(server) as client:
                result = await client.call_tool(_name, kwargs)
            return _tool_payload(result)

        tools.append(
            StructuredTool.from_function(
                coroutine=_call,
                name=spec.name,
                description=spec.description or spec.name,
                infer_schema=False,
                args_schema=_args_from_mcp_schema(spec.name, spec.inputSchema),
            )
        )
    return tools


def _args_from_mcp_schema(tool_name: str, schema: dict | None):
    """Build a tiny Pydantic model from MCP JSON schema so the LLM can call the tool."""
    from pydantic import create_model

    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    fields = {}
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
    for key, spec in props.items():
        py_type = type_map.get(spec.get("type", "string"), str)
        default = ... if key in required else spec.get("default", None)
        fields[key] = (py_type, default)
    if not fields:
        fields["input"] = (str, ...)
    return create_model(f"{tool_name}_Args", **fields)


def _split_tools(tools):
    sql, rag, all_tools = [], [], []
    for tool in tools:
        all_tools.append(tool)
        if tool.name in SQL_TOOL_NAMES:
            sql.append(tool)
        elif tool.name in RAG_TOOL_NAMES:
            rag.append(tool)
        else:
            sql.append(tool)
            rag.append(tool)
    return sql, rag, all_tools


ROUTER_PROMPT = """You route questions for a customer-support assistant.
Choose one:
- sql: customer profiles, tickets, plans, account status, anything in the company database
- rag: company policies (refund, privacy, SLA) from PDF documents
- hybrid: needs BOTH a specific customer AND a policy (example: does this customer's ticket match refund policy)
Question: {question}
"""

SQL_PROMPT = (
    "You are the SQL specialist for Acme Support. "
    "Use MCP tools to look up customers and tickets. "
    "Never invent customers. If not found, say so. "
    "Write a short, clear answer for a support agent named John."
)

RAG_PROMPT = (
    "You are the policy specialist for Acme Support. "
    "Use search_policies to read PDF chunks. "
    "Cite the source filename. Do not invent policy. "
    "Write a short, clear answer for a support agent named John."
)

HYBRID_PROMPT = (
    "You are a support specialist. Use customer/ticket tools AND search_policies. "
    "Combine the customer facts with the relevant policy. "
    "Cite policy source filenames. Write a short answer for John."
)


_app = None


async def build_graph():
    llm = get_llm()
    tools = await _mcp_tools()
    sql_tools, rag_tools, all_tools = _split_tools(tools)

    sql_agent = create_agent(llm, sql_tools, system_prompt=SQL_PROMPT, name="sql_agent")
    rag_agent = create_agent(llm, rag_tools, system_prompt=RAG_PROMPT, name="rag_agent")
    hybrid_agent = create_agent(llm, all_tools, system_prompt=HYBRID_PROMPT, name="hybrid_agent")
    router = llm.with_structured_output(RouteDecision)

    def last_user_text(state: AgentState) -> str:
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
                return str(message.content)
        return str(state["messages"][-1].content)

    def supervisor(state: AgentState):
        question = last_user_text(state)
        decision = router.invoke(ROUTER_PROMPT.format(question=question))
        route = decision.next if decision else "hybrid"
        return {"route": route}

    async def run_sql(state: AgentState):
        result = await sql_agent.ainvoke({"messages": state["messages"]})
        return {"messages": [result["messages"][-1]]}

    async def run_rag(state: AgentState):
        result = await rag_agent.ainvoke({"messages": state["messages"]})
        return {"messages": [result["messages"][-1]]}

    async def run_hybrid(state: AgentState):
        result = await hybrid_agent.ainvoke({"messages": state["messages"]})
        return {"messages": [result["messages"][-1]]}

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("sql", run_sql)
    graph.add_node("rag", run_rag)
    graph.add_node("hybrid", run_hybrid)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        {"sql": "sql", "rag": "rag", "hybrid": "hybrid"},
    )
    graph.add_edge("sql", END)
    graph.add_edge("rag", END)
    graph.add_edge("hybrid", END)
    return graph.compile()


async def get_app():
    global _app
    if _app is None:
        _app = await build_graph()
    return _app


async def ask(question: str) -> dict:
    app = await get_app()
    result = await app.ainvoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content
    return {"route": result.get("route", ""), "answer": answer}


def ask_sync(question: str) -> dict:
    """Run the async graph from Streamlit or other sync callers."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(ask(question))).result()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]).strip() or "What is the current refund policy?"
    print(f"Q: {question}")
    output = asyncio.run(ask(question))
    print(f"route: {output['route']}")
    print(output["answer"])


if __name__ == "__main__":
    main()
