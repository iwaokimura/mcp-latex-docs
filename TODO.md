# TODO

Future improvements to consider, roughly in order of priority.

## Search quality

- **Custom `\newtheorem` support** — the parser currently recognizes a fixed set of environment names. Documents that define custom environments via `\newtheorem{mytheorem}{My Theorem}` in the preamble should have those environments extracted automatically.
- **Tune definition boost** — the `_DEF_BOOST = 1.15` multiplier in `store.py` was chosen conservatively. Evaluate on a real corpus and adjust.
- **Cross-lingual search** — a query in Japanese should surface results from English documents and vice versa. Evaluate whether the current e5 model handles this well enough or if explicit cross-lingual re-ranking is needed.

## Indexing

- **Incremental reindex** — `index_folder` currently re-embeds every block on every call. Add mtime tracking so only new or modified `.tex` files are reprocessed.
- **Progress reporting** — large document collections (50+ files) make `index_folder` slow with no feedback. Emit MCP progress notifications during parsing and embedding so Perplexity can show a progress indicator.
- **Preamble-aware parsing** — extract `\newtheorem` declarations from the document preamble and register them dynamically with pylatexenc before parsing the body.

## Server

- **Configurable DB path** — expose the ChromaDB storage directory as an environment variable (`MCP_LATEX_DOCS_DB`) so users can control where data is stored without editing code.
- **Startup model preload** — add an MCP `lifespan` hook that loads both embedding models at server startup rather than on the first tool call, avoiding a timeout on the initial `index_folder`.

## Robustness

- **Large file handling** — blocks longer than 512 tokens are silently truncated by the tokenizers. Detect oversized blocks and split them at sentence boundaries before embedding.
- **Error reporting per file** — if one `.tex` file fails to parse (e.g. broken LaTeX), `index_folder` should continue with the remaining files and report which files failed in the summary.

## Testing

- **Japanese document fixture** — add a sample `.tex` file written in Japanese to `tests/fixtures/` and extend the smoke test to cover Japanese prose search.
- **Integration test for `search`** — the current store tests use fake embeddings. Add at least one test that loads real (small) models and checks that semantically similar queries return the expected blocks.
