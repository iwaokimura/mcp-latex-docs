"""
server.py - MCP stdio server for mcp-latex-docs.

Exposes five tools to Perplexity (or any MCP client):
  index_folder   - parse and embed all .tex files in a folder
  search         - semantic search over indexed blocks
  get_by_label   - exact lookup by LaTeX \\label
  list_folders   - list indexed folders with counts
  remove_folder  - delete all indexed blocks for a folder
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any

import mcp.server as mcp_server
import mcp.types as types
from mcp.server.stdio import stdio_server

from mcp_latex_docs.embedder import Embedder
from mcp_latex_docs.parser import parse_folder
from mcp_latex_docs.store import Store

# ---------------------------------------------------------------------------
# Global singletons (created once, reused across tool calls)
# ---------------------------------------------------------------------------

_embedder: Embedder | None = None
_store:    Store    | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

server = mcp_server.Server("mcp-latex-docs")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="index_folder",
            description=(
                "Parse all LaTeX (.tex) files in a folder and index them into "
                "the vector database. Re-running updates existing entries. "
                "Returns a summary of how many files and blocks were indexed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_path": {
                        "type": "string",
                        "description": "Absolute path to the folder containing .tex files.",
                    },
                },
                "required": ["folder_path"],
            },
        ),
        types.Tool(
            name="search",
            description=(
                "Semantic search over indexed LaTeX blocks (theorems, lemmas, "
                "definitions, proofs, etc.). The query can be natural language, "
                "LaTeX math notation, or a mix of both."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Can be natural language "
                            "('definition of compactness'), LaTeX "
                            "('$\\\\phi_{foobar}$'), or mixed."
                        ),
                    },
                    "folder_path": {
                        "type": "string",
                        "description": "Restrict search to this folder (optional).",
                    },
                    "env_type": {
                        "type": "string",
                        "description": (
                            "Restrict to a specific environment type: "
                            "theorem, lemma, definition, proof, proposition, "
                            "corollary, remark, example (optional)."
                        ),
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_by_label",
            description=(
                "Retrieve a specific block by its LaTeX \\label{...} value. "
                "Returns the exact block or nothing if not found."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The LaTeX label string, e.g. 'thm:main' or 'def:uniform-conv'.",
                    },
                    "folder_path": {
                        "type": "string",
                        "description": "Restrict lookup to this folder (optional).",
                    },
                },
                "required": ["label"],
            },
        ),
        types.Tool(
            name="list_folders",
            description="List all indexed folders with their file and block counts.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="remove_folder",
            description=(
                "Remove all indexed blocks for a given folder from the database. "
                "Returns the number of blocks deleted."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_path": {
                        "type": "string",
                        "description": "Absolute path to the folder to remove.",
                    },
                },
                "required": ["folder_path"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _dispatch, name, arguments
        )
        return [types.TextContent(type="text", text=result)]
    except Exception:
        error_text = traceback.format_exc()
        return [types.TextContent(type="text", text=f"Error:\n{error_text}")]


def _dispatch(name: str, args: dict[str, Any]) -> str:
    if name == "index_folder":
        return _tool_index_folder(args["folder_path"])
    if name == "search":
        return _tool_search(
            query=args["query"],
            folder_path=args.get("folder_path"),
            env_type=args.get("env_type"),
            n_results=int(args.get("n_results", 5)),
        )
    if name == "get_by_label":
        return _tool_get_by_label(
            label=args["label"],
            folder_path=args.get("folder_path"),
        )
    if name == "list_folders":
        return _tool_list_folders()
    if name == "remove_folder":
        return _tool_remove_folder(args["folder_path"])
    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_index_folder(folder_path: str) -> str:
    blocks = parse_folder(folder_path)
    if not blocks:
        return f"No theorem-like blocks found in {folder_path}."

    embedder  = _get_embedder()
    store     = _get_store()
    embeddings = embedder.embed_blocks(blocks)
    store.upsert(blocks, embeddings)

    file_set = {b.source_file for b in blocks}
    return (
        f"Indexed {len(file_set)} file(s), {len(blocks)} block(s) "
        f"from {folder_path}."
    )


def _tool_search(
    query: str,
    folder_path: str | None,
    env_type: str | None,
    n_results: int,
) -> str:
    store    = _get_store()
    embedder = _get_embedder()

    query_emb = embedder.embed_query(query)
    results   = store.search(
        query_emb=query_emb,
        query_text=query,
        folder_path=folder_path,
        env_type=env_type,
        n_results=n_results,
    )

    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        label_str   = f"  label:   {r.label}"   if r.label   else ""
        section_str = f"  section: {r.section}"  if r.section else ""
        lines = [
            f"[{i}] {r.env_type.upper()}  (score: {r.score:.3f}, view: {r.matched_view})",
            f"  file: {r.source_file}",
        ]
        if section_str:
            lines.append(section_str)
        if label_str:
            lines.append(label_str)
        lines.append(f"  ---\n{r.text.strip()}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _tool_get_by_label(label: str, folder_path: str | None) -> str:
    store  = _get_store()
    result = store.get_by_label(label, folder_path=folder_path)

    if result is None:
        scope = f" in {folder_path}" if folder_path else ""
        return f"No block with label '{label}' found{scope}."

    lines = [
        f"{result.env_type.upper()}  (label: {result.label})",
        f"  file: {result.source_file}",
    ]
    if result.section:
        lines.append(f"  section: {result.section}")
    lines.append(f"  ---\n{result.text.strip()}")
    return "\n".join(lines)


def _tool_list_folders() -> str:
    store   = _get_store()
    folders = store.list_folders()

    if not folders:
        return "No folders indexed yet."

    lines = ["Indexed folders:"]
    for f in folders:
        lines.append(
            f"  {f.folder}  ({f.file_count} file(s), {f.block_count} block(s))"
        )
    return "\n".join(lines)


def _tool_remove_folder(folder_path: str) -> str:
    store   = _get_store()
    removed = store.remove_folder(folder_path)

    if removed == 0:
        return f"No indexed blocks found for {folder_path}."
    return f"Removed {removed} block(s) from {folder_path}."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
