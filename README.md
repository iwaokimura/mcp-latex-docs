# mcp-latex-docs

A local [MCP](https://modelcontextprotocol.io/) server that turns your LaTeX math documents into a searchable semantic database.  Works with any MCP-compatible client — tested with the [Perplexity Mac app](https://perplexity.ai) and [GitHub Copilot in VS Code](https://code.visualstudio.com/).

## What it does

Mathematical documents written in LaTeX are structured around named environments — `theorem`, `lemma`, `definition`, `proof`, and so on. This server parses those documents, recognizes that structure, embeds each block using a pair of neural models (one for multilingual prose, one for mathematical notation), and stores everything in a local [ChromaDB](https://www.trychroma.com/) vector database.

Once indexed, you can ask your AI assistant questions like:

- *"Read all math documents in folder `/Users/me/papers`"*
- *"Search the definition of uniform continuity"*
- *"Find theorems about compact sets $K \subseteq \mathbb{R}^n$"*
- *"Get the block with label `thm:heine-borel`"*

Queries can be plain English, raw LaTeX, or a mix of both. Japanese documents are supported.

## Architecture

```
.tex files
    │
    ▼
pylatexenc parser
    └── extracts theorem / lemma / definition / proof / ...  blocks
    └── captures \label{}, section title, source file
    └── resolves \input{} / \include{} recursively
    └── detects file encoding (UTF-8, EUC-JP, Shift-JIS)
    │
    ▼
Two embedding models (Apple Silicon / MPS)
    ├── intfloat/multilingual-e5-large-instruct  → prose + Japanese
    └── witiko/mathberta                          → LaTeX math notation
    │   (blocks > 512 tokens are split at natural boundaries before embedding)
    │
    ▼
ChromaDB (local persistent store)
    ├── text_view   — e5 embeddings
    └── math_view   — mathberta embeddings

Query routing (automatic)
    ├── plain text query  → text_view only
    ├── LaTeX-only query  → math_view only
    └── mixed query       → both views, results merged and ranked
```

## Requirements

- macOS with Apple Silicon
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
git clone git@github.com:iwaokimura/mcp-latex-docs.git
cd mcp-latex-docs
uv pip install -e .
```

The two embedding models (~1 GB total) are downloaded from Hugging Face on the first `index_folder` call and cached in `~/.cache/huggingface/hub/`.

## Connecting to Perplexity Mac app

In the Perplexity Mac app, open **Settings → MCP Servers** and add a new local server with:

```json
{
  "command": "/path/to/mcp-latex-docs/.venv/bin/mcp-latex-docs"
}
```

Replace `/path/to/mcp-latex-docs` with the absolute path to your clone. Then grant Perplexity **Full Disk Access** in **System Settings → Privacy & Security → Full Disk Access** so the server can read your document folders.

## Connecting to GitHub Copilot in VS Code

Create (or edit) `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "mcp-latex-docs": {
      "type": "stdio",
      "command": "/path/to/mcp-latex-docs/.venv/bin/python3",
      "args": ["-m", "mcp_latex_docs.server"]
    }
  }
}
```

Replace `/path/to/mcp-latex-docs` with the absolute path to your clone. The server's tools then become available in Copilot Chat agent mode — you can ask Copilot to index a folder of papers and search them alongside your codebase.

## Available tools

| Tool | Description |
|---|---|
| `index_folder(folder_path)` | Parse and embed all `.tex` files in a folder. Re-running updates existing entries. |
| `search(query, folder_path?, env_type?, n_results?)` | Semantic search. Query can be natural language, LaTeX, or mixed. |
| `get_by_label(label, folder_path?)` | Exact lookup by `\label{...}` value. |
| `list_folders()` | List all indexed folders with file and block counts. |
| `remove_folder(folder_path)` | Remove all indexed blocks for a folder. |

### Supported environments

`theorem`, `lemma`, `proposition`, `corollary`, `definition`, `proof`, `remark`, `example`, `conjecture`, `claim`, `notation`, `observation`
(and their common aliases: `thm`, `lem`, `prop`, `cor`, `defn`, `dfn`, `rem`, `ex`, `conj`)

## Development

```bash
uv run pytest          # run all tests (no GPU required)
uv run python smoke_test.py   # end-to-end test with real models
```

The project follows a four-module layout:

```
src/mcp_latex_docs/
├── parser.py    — LaTeX block extractor
├── embedder.py  — dual-model embedder, block splitting, and query routing
├── store.py     — ChromaDB upsert, search, and ranking
└── server.py    — MCP stdio server
```

Vector data is stored in `~/.mcp-latex-docs/chroma/` and persists across server restarts.

## License

MIT
