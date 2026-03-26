# mcp-latex-docs — Design Memo

## Overview

A local MCP server for the Perplexity Mac app that parses mathematical documents written in LaTeX, builds a semantic vector database, and answers queries via stdio transport.

**Typical usage:**
- "Read all math documents in folder Foo"
- "Search the definition of the term foobar"
- "Search the definition of the symbol $\phi_{foobar}$"

---

## Environment

- **Platform:** macOS, Apple Silicon (MPS available for inference)
- **Package manager:** `uv` (flat layout)
- **Python:** latest available via uv

---

## Dependencies

| Package | Purpose |
|---|---|
| `pylatexenc` | LaTeX parsing |
| `chromadb` | Local persistent vector store |
| `sentence-transformers` | multilingual-e5-large-instruct embeddings |
| `transformers` + `torch` | witiko/mathberta embeddings (MPS backend) |
| `mcp` | Anthropic MCP Python SDK (stdio transport) |
| `chardet` | File encoding detection (for Japanese docs) |

---

## LaTeX Parsing

**Library:** `pylatexenc`

### Extracted environments

| Environment | Aliases |
|---|---|
| `definition` | `defn`, `dfn` |
| `theorem` | `thm` |
| `lemma` | `lem` |
| `proposition` | `prop` |
| `proof` | — |
| `corollary` | `cor` |
| `remark` | `rem` |
| `example` | `ex` |

### Per-block metadata extracted

- `env_type` — normalized environment name (e.g. `theorem`)
- `label` — from `\label{...}` inside the block, if present
- `section` — enclosing section/subsection title
- `source_file` — absolute path to the `.tex` file
- `folder` — absolute path to the indexed folder
- `block_id` — stable unique ID: `"<filename>::<env_type>::<index>"`

### Multi-file support

`\input{}` and `\include{}` are resolved recursively within the indexed folder.

### Language support

- English and Japanese documents supported
- File encoding auto-detected via `chardet` (handles UTF-8, EUC-JP, Shift-JIS)
- pylatexenc works on Python Unicode strings; encoding is resolved at file read time
- Japanese prose is passed through as-is (Unicode); LaTeX math notation is language-agnostic

---

## Embedding Models

Two models are used. **Every extracted block is embedded with both models** and stored in two separate ChromaDB collections.

### Model 1 — Text (prose, multilingual)

- **Model:** `intfloat/multilingual-e5-large-instruct`
- **Size:** 0.6B parameters
- **Dimension:** 1,024
- **Context:** 512 tokens
- **Languages:** 100+ including Japanese and English
- **Use:** Embeds the full block text (prose + inline math as raw LaTeX strings)
- **Collection:** `text_view`

### Model 2 — Math (formulae + mixed content)

- **Model:** `witiko/mathberta`
- **Base:** RoBERTa-base with extended LaTeX tokenizer
- **Dimension:** 768
- **Context:** 512 tokens
- **Training data:** ArXMLiv 2020 (1.58M ArXiv papers) + Math StackExchange (2.47M Q&A)
- **Use:** Embeds the full block text; understands LaTeX math notation semantically
- **Collection:** `math_view`

### Why embed every block with both models

A theorem block typically contains interleaved prose and inline math (e.g. "Let $f$ be a function such that $\int_0^1 f(x)\,dx = 0$"). Splitting prose and math would break semantic continuity, so the whole block is embedded by each model. The two collections provide complementary views: one optimized for language semantics, one for mathematical notation.

---

## ChromaDB Schema

Two collections: `text_view` and `math_view`.

Each entry stores:
- **document:** raw block text (LaTeX source of the environment body)
- **embedding:** model-specific vector
- **metadata:**
  ```
  folder:      "/absolute/path/to/Foo"
  source_file: "/absolute/path/to/Foo/main.tex"
  env_type:    "theorem"
  label:       "thm:compactness"     # empty string if absent
  section:     "3.2 Compactness"     # empty string if unknown
  block_id:    "main.tex::theorem::3"
  ```

`block_id` is used as the ChromaDB document ID, enabling idempotent upserts (re-indexing a file updates existing entries rather than duplicating them).

---

## MCP Tools

### `index_folder`

```
index_folder(folder_path: str) -> str
```

Recursively scans `folder_path` for `.tex` files, parses each one, embeds all extracted blocks with both models, and upserts into ChromaDB.

Returns a summary string: `"Indexed 4 files, 87 blocks from /path/to/Foo"`.

---

### `search`

```
search(
    query:       str,
    folder_path: str | None = None,
    env_type:    str | None = None,
    n_results:   int        = 5
) -> list[SearchResult]
```

Semantic search over indexed blocks.

**Query routing** (automatic, based on query content):

| Query content | Models used | Collections searched |
|---|---|---|
| Natural language only | multilingual-e5 | `text_view` |
| LaTeX only (`$...$`, `\[...\]`, `\cmd`) | mathberta | `math_view` |
| Mixed | Both | Both, results merged |

**Ranking:**
- Results from both collections are merged and deduplicated by `block_id`
- Scores are normalized before merging
- Definition blocks (`env_type = "definition"`) receive a rank boost when the query contains "definition of"

**Filters:**
- `folder_path`: scopes search to one indexed folder
- `env_type`: restricts to a specific environment type

---

### `get_by_label`

```
get_by_label(label: str, folder_path: str | None = None) -> SearchResult | None
```

Exact metadata lookup by `\label{}` value. No embedding used. Returns the block or `None`.

---

### `list_folders`

```
list_folders() -> list[FolderInfo]
```

Returns all indexed folders with file count and block count.

---

### `remove_folder`

```
remove_folder(folder_path: str) -> str
```

Deletes all indexed blocks whose metadata `folder` matches `folder_path` from both collections.

---

## Transport

- **Protocol:** MCP stdio (Anthropic MCP Python SDK)
- **Client:** Perplexity Mac app ("local MCP server" feature)

---

## File Layout

```
mcp-latex-docs/
├── pyproject.toml
├── DESIGN.md
├── src/
│   └── mcp_latex_docs/
│       ├── __init__.py
│       ├── server.py      # MCP stdio server, tool definitions
│       ├── parser.py      # pylatexenc block extractor
│       ├── embedder.py    # both models, query routing logic
│       └── store.py       # ChromaDB upsert / search / filter
└── tests/
```

---

## Implementation Order

1. `parser.py` — block extraction, metadata, multi-file resolution, encoding detection
2. `embedder.py` — model loading (MPS device), embed functions, query type detection
3. `store.py` — ChromaDB init, upsert, search, merge/rank, label lookup
4. `server.py` — MCP tool wiring, stdio transport
