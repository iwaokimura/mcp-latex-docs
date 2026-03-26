"""
smoke_test.py - End-to-end integration test with real models.

Run with:  uv run python smoke_test.py
"""

import sys
import textwrap
from pathlib import Path

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "sample.tex"
DB_PATH = Path(__file__).parent / ".smoke-chroma"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def show(label: str, text: str) -> None:
    print(f"\n[{label}]")
    print(textwrap.indent(text.strip(), "  "))


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Parser
    # ------------------------------------------------------------------ #
    section("1. Parser")
    from mcp_latex_docs.parser import parse_folder
    blocks = parse_folder(FIXTURE.parent)
    print(f"  Found {len(blocks)} blocks in {FIXTURE.name}")
    for b in blocks:
        label = f" [{b.label}]" if b.label else ""
        print(f"    {b.env_type:12s}{label}")

    # ------------------------------------------------------------------ #
    # 2. Embedder (loads models — slow on first run)
    # ------------------------------------------------------------------ #
    section("2. Embedder  (downloading models if needed…)")
    from mcp_latex_docs.embedder import Embedder
    embedder = Embedder()
    print("  Embedding blocks with both models…")
    embeddings = embedder.embed_blocks(blocks)
    e0 = embeddings[0]
    print(f"  text_vec dim : {len(e0.text_vec)}")
    print(f"  math_vec dim : {len(e0.math_vec)}")
    print(f"  block_id     : {e0.block_id}")

    # ------------------------------------------------------------------ #
    # 3. Store — upsert
    # ------------------------------------------------------------------ #
    section("3. Store — upsert")
    from mcp_latex_docs.store import Store
    store = Store(db_path=DB_PATH)
    store.upsert(blocks, embeddings)
    folders = store.list_folders()
    for f in folders:
        print(f"  {f.folder}  ({f.file_count} file, {f.block_count} blocks)")

    # ------------------------------------------------------------------ #
    # 4. Search — natural language (TEXT query)
    # ------------------------------------------------------------------ #
    section("4. Search — natural language")
    query = "definition of uniform continuity"
    print(f"  query: \"{query}\"")
    qe = embedder.embed_query(query)
    print(f"  query type: {qe.query_type.name}")
    results = store.search(qe, query_text=query, n_results=3)
    for r in results:
        print(f"\n  [{r.env_type}] score={r.score:.3f}  view={r.matched_view}")
        if r.label:
            print(f"  label: {r.label}")
        print(f"  {r.text.strip()[:120]}…")

    # ------------------------------------------------------------------ #
    # 5. Search — LaTeX math (MATH query)
    # ------------------------------------------------------------------ #
    section("5. Search — LaTeX math")
    query = r"\int_a^b f(x)\, dx"
    print(f"  query: \"{query}\"")
    qe = embedder.embed_query(query)
    print(f"  query type: {qe.query_type.name}")
    results = store.search(qe, query_text=query, n_results=3)
    for r in results:
        print(f"\n  [{r.env_type}] score={r.score:.3f}  view={r.matched_view}")
        print(f"  {r.text.strip()[:120]}…")

    # ------------------------------------------------------------------ #
    # 6. Search — mixed query
    # ------------------------------------------------------------------ #
    section("6. Search — mixed query")
    query = r"theorems about compact sets $K \subseteq \mathbb{R}^n$"
    print(f"  query: \"{query}\"")
    qe = embedder.embed_query(query)
    print(f"  query type: {qe.query_type.name}")
    results = store.search(qe, query_text=query, n_results=3)
    for r in results:
        print(f"\n  [{r.env_type}] score={r.score:.3f}  view={r.matched_view}")
        print(f"  {r.text.strip()[:120]}…")

    # ------------------------------------------------------------------ #
    # 7. get_by_label
    # ------------------------------------------------------------------ #
    section("7. get_by_label")
    for lbl in ["thm:heine-borel", "def:riemann-integral", "nonexistent"]:
        result = store.get_by_label(lbl)
        if result:
            print(f"  {lbl!r:30s}  → [{result.env_type}]  score={result.score}")
        else:
            print(f"  {lbl!r:30s}  → not found")

    # ------------------------------------------------------------------ #
    # 8. env_type filter
    # ------------------------------------------------------------------ #
    section("8. Search with env_type filter (definition only)")
    query = "continuity"
    qe = embedder.embed_query(query)
    results = store.search(qe, query_text=query, env_type="definition", n_results=5)
    print(f"  query: \"{query}\",  env_type=definition")
    for r in results:
        print(f"  [{r.env_type}] score={r.score:.3f}  label={r.label or '—'}")

    # ------------------------------------------------------------------ #
    # 9. Cleanup
    # ------------------------------------------------------------------ #
    section("9. Cleanup")
    removed = store.remove_folder(str(FIXTURE.parent.resolve()))
    print(f"  Removed {removed} blocks.")
    import shutil
    shutil.rmtree(DB_PATH, ignore_errors=True)
    print("  Smoke-test DB deleted.")

    section("ALL DONE")


if __name__ == "__main__":
    main()
