"""
store.py - ChromaDB persistence layer.

Two collections:
  text_view  — embeddings from multilingual-e5-large-instruct
  math_view  — embeddings from witiko/mathberta

Every block is stored in both collections.
Search results from both are merged, deduplicated, and ranked.
Definition blocks receive a score boost when the query requests a definition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from mcp_latex_docs.embedder import BlockEmbedding, QueryEmbedding, QueryType
from mcp_latex_docs.parser import Block

# ---------------------------------------------------------------------------
# Default storage path
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path.home() / ".mcp-latex-docs" / "chroma"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    block_id:    str
    env_type:    str
    label:       str
    section:     str
    source_file: str
    folder:      str
    text:        str
    score:       float
    matched_view: str   # "text", "math", or "both"


@dataclass
class FolderInfo:
    folder:     str
    file_count: int
    block_count: int


# ---------------------------------------------------------------------------
# Definition-query boost
# ---------------------------------------------------------------------------

_DEF_QUERY_RE = re.compile(
    r"\b(definition|define|defined|meaning)\b",
    re.IGNORECASE,
)
_DEF_BOOST = 1.15   # multiply score of definition blocks by this factor


def _should_boost_definitions(query: str) -> bool:
    return bool(_DEF_QUERY_RE.search(query))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class Store:
    """
    Persistent vector store backed by ChromaDB.

    Parameters
    ----------
    db_path : path-like, optional
        Directory for ChromaDB data.  Defaults to ~/.mcp-latex-docs/chroma.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(path))
        cos = {"hnsw:space": "cosine"}
        self._text_col = self._client.get_or_create_collection(
            "text_view", metadata=cos
        )
        self._math_col = self._client.get_or_create_collection(
            "math_view", metadata=cos
        )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(self, blocks: list[Block], embeddings: list[BlockEmbedding]) -> None:
        """Store blocks and their embeddings into both collections."""
        if not blocks:
            return

        emb_by_id = {e.block_id: e for e in embeddings}

        ids, docs, metas, text_vecs, math_vecs = [], [], [], [], []
        for b in blocks:
            emb = emb_by_id.get(b.block_id)
            if emb is None:
                continue
            ids.append(b.block_id)
            docs.append(b.text)
            metas.append({
                "env_type":    b.env_type,
                "label":       b.label,
                "section":     b.section,
                "source_file": b.source_file,
                "folder":      b.folder,
            })
            text_vecs.append(emb.text_vec)
            math_vecs.append(emb.math_vec)

        if not ids:
            return

        self._text_col.upsert(
            ids=ids, documents=docs, metadatas=metas, embeddings=text_vecs
        )
        self._math_col.upsert(
            ids=ids, documents=docs, metadatas=metas, embeddings=math_vecs
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_emb: QueryEmbedding,
        query_text: str = "",
        folder_path: str | None = None,
        env_type: str | None = None,
        n_results: int = 5,
    ) -> list[SearchResult]:
        """
        Semantic search over stored blocks.

        Searches text_view, math_view, or both depending on query type,
        then merges and ranks the results.
        """
        where = _build_where(folder_path, env_type)
        fetch = n_results * 2   # fetch more candidates before merging

        raw: dict[str, dict] = {}   # block_id -> {meta, doc, scores}

        if query_emb.text_vec is not None:
            self._query_collection(
                self._text_col, query_emb.text_vec, where, fetch, "text", raw
            )
        if query_emb.math_vec is not None:
            self._query_collection(
                self._math_col, query_emb.math_vec, where, fetch, "math", raw
            )

        boost_defs = _should_boost_definitions(query_text)
        results = _merge_and_rank(raw, boost_defs, n_results)
        return results

    def _query_collection(
        self,
        col,
        vec: list[float],
        where: dict | None,
        n: int,
        view_name: str,
        acc: dict,
    ) -> None:
        """Run a single-collection query and accumulate results into *acc*."""
        kwargs: dict = {
            "query_embeddings": [vec],
            "n_results": min(n, col.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            res = col.query(**kwargs)
        except Exception:
            return

        ids       = res["ids"][0]
        docs      = res["documents"][0]
        metas     = res["metadatas"][0]
        distances = res["distances"][0]

        for bid, doc, meta, dist in zip(ids, docs, metas, distances):
            # cosine distance in [0,2]; score = 1 - dist ∈ [-1, 1]
            score = 1.0 - dist
            if bid not in acc:
                acc[bid] = {
                    "doc":    doc,
                    "meta":   meta,
                    "scores": {},
                }
            acc[bid]["scores"][view_name] = score

    # ------------------------------------------------------------------
    # Label lookup
    # ------------------------------------------------------------------

    def get_by_label(
        self,
        label: str,
        folder_path: str | None = None,
    ) -> SearchResult | None:
        """Exact metadata lookup by \\label value. No embedding used."""
        where: dict = {"label": {"$eq": label}}
        if folder_path:
            where = {"$and": [where, {"folder": {"$eq": folder_path}}]}

        res = self._text_col.get(
            where=where,
            include=["documents", "metadatas"],
            limit=1,
        )
        if not res["ids"]:
            return None

        bid  = res["ids"][0]
        doc  = res["documents"][0]
        meta = res["metadatas"][0]
        return SearchResult(
            block_id=bid,
            env_type=meta.get("env_type", ""),
            label=meta.get("label", ""),
            section=meta.get("section", ""),
            source_file=meta.get("source_file", ""),
            folder=meta.get("folder", ""),
            text=doc,
            score=1.0,
            matched_view="label",
        )

    # ------------------------------------------------------------------
    # Folder management
    # ------------------------------------------------------------------

    def list_folders(self) -> list[FolderInfo]:
        """Return all indexed folders with file and block counts."""
        res = self._text_col.get(include=["metadatas"])
        if not res["ids"]:
            return []

        folder_files:  dict[str, set[str]] = {}
        folder_blocks: dict[str, int]      = {}

        for meta in res["metadatas"]:
            fld  = meta.get("folder", "")
            ffile = meta.get("source_file", "")
            folder_files.setdefault(fld, set()).add(ffile)
            folder_blocks[fld] = folder_blocks.get(fld, 0) + 1

        return [
            FolderInfo(
                folder=fld,
                file_count=len(folder_files[fld]),
                block_count=folder_blocks[fld],
            )
            for fld in sorted(folder_files)
        ]

    def remove_folder(self, folder_path: str) -> int:
        """Delete all blocks whose folder metadata matches *folder_path*."""
        where = {"folder": {"$eq": folder_path}}

        # Collect IDs first (delete by where is supported but count is useful)
        res = self._text_col.get(where=where, include=[])
        count = len(res["ids"])
        if count == 0:
            return 0

        self._text_col.delete(where=where)
        self._math_col.delete(where=where)
        return count


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_where(
    folder_path: str | None,
    env_type: str | None,
) -> dict | None:
    clauses = []
    if folder_path:
        clauses.append({"folder":   {"$eq": folder_path}})
    if env_type:
        clauses.append({"env_type": {"$eq": env_type}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _merge_and_rank(
    raw: dict[str, dict],
    boost_definitions: bool,
    n_results: int,
) -> list[SearchResult]:
    results = []
    for bid, data in raw.items():
        scores = data["scores"]
        meta   = data["meta"]
        doc    = data["doc"]

        # Combined score: max across views (both views reward, not penalize)
        base_score   = max(scores.values())
        matched_view = "both" if len(scores) > 1 else next(iter(scores))

        # Definition boost
        if boost_definitions and meta.get("env_type") == "definition":
            base_score = min(base_score * _DEF_BOOST, 1.0)

        results.append(SearchResult(
            block_id=bid,
            env_type=meta.get("env_type", ""),
            label=meta.get("label", ""),
            section=meta.get("section", ""),
            source_file=meta.get("source_file", ""),
            folder=meta.get("folder", ""),
            text=doc,
            score=base_score,
            matched_view=matched_view,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:n_results]
