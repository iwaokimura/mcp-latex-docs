"""Tests for store.py — uses an in-memory ChromaDB client via tmp_path."""

import pytest

import chromadb
from mcp_latex_docs.embedder import BlockEmbedding, QueryEmbedding, QueryType
from mcp_latex_docs.parser import Block
from mcp_latex_docs.store import (
    Store,
    SearchResult,
    _build_where,
    _merge_and_rank,
    _should_boost_definitions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "chroma")


def _block(
    env_type: str,
    label: str = "",
    text: str = "some text",
    folder: str = "/foo",
    source_file: str = "/foo/main.tex",
    index: int = 1,
) -> Block:
    return Block(
        env_type=env_type,
        text=text,
        label=label,
        section="Section 1",
        source_file=source_file,
        folder=folder,
        block_id=f"main.tex::{env_type}::{index}",
    )


def _emb(block: Block, dim_text: int = 8, dim_math: int = 6) -> BlockEmbedding:
    """Deterministic fake embeddings (unit vectors)."""
    import hashlib, math
    h = int(hashlib.md5(block.block_id.encode()).hexdigest(), 16)
    tv = [(((h >> i) & 1) * 2 - 1) / math.sqrt(dim_text) for i in range(dim_text)]
    mv = [(((h >> i) & 1) * 2 - 1) / math.sqrt(dim_math) for i in range(dim_math)]
    return BlockEmbedding(block_id=block.block_id, text_vec=tv, math_vec=mv)


def _query_emb(
    text_vec: list[float] | None = None,
    math_vec: list[float] | None = None,
    qtype: QueryType = QueryType.TEXT,
) -> QueryEmbedding:
    return QueryEmbedding(query_type=qtype, text_vec=text_vec, math_vec=math_vec)


# ---------------------------------------------------------------------------
# upsert / basic retrieval
# ---------------------------------------------------------------------------

def test_upsert_and_count(store):
    blocks = [_block("theorem"), _block("definition", index=2)]
    embs   = [_emb(b) for b in blocks]
    store.upsert(blocks, embs)
    assert store._text_col.count() == 2
    assert store._math_col.count() == 2


def test_upsert_idempotent(store):
    b = _block("lemma")
    e = _emb(b)
    store.upsert([b], [e])
    store.upsert([b], [e])   # second upsert of same block_id
    assert store._text_col.count() == 1


def test_upsert_empty(store):
    store.upsert([], [])
    assert store._text_col.count() == 0


# ---------------------------------------------------------------------------
# get_by_label
# ---------------------------------------------------------------------------

def test_get_by_label_found(store):
    b = _block("theorem", label="thm:main")
    store.upsert([b], [_emb(b)])
    result = store.get_by_label("thm:main")
    assert result is not None
    assert result.label    == "thm:main"
    assert result.env_type == "theorem"
    assert result.score    == 1.0


def test_get_by_label_not_found(store):
    assert store.get_by_label("nonexistent") is None


def test_get_by_label_folder_scoped(store):
    b1 = _block("theorem", label="thm:x", folder="/foo", index=1)
    b2 = _block("theorem", label="thm:x", folder="/bar",
                source_file="/bar/main.tex", index=2)
    store.upsert([b1, b2], [_emb(b1), _emb(b2)])

    r = store.get_by_label("thm:x", folder_path="/foo")
    assert r is not None
    assert r.folder == "/foo"


# ---------------------------------------------------------------------------
# list_folders / remove_folder
# ---------------------------------------------------------------------------

def test_list_folders(store):
    blocks = [
        _block("theorem", folder="/proj/A", source_file="/proj/A/a.tex", index=1),
        _block("lemma",   folder="/proj/A", source_file="/proj/A/a.tex", index=2),
        _block("theorem", folder="/proj/B", source_file="/proj/B/b.tex", index=3),
    ]
    store.upsert(blocks, [_emb(b) for b in blocks])

    folders = store.list_folders()
    assert len(folders) == 2
    by_folder = {f.folder: f for f in folders}
    assert by_folder["/proj/A"].block_count == 2
    assert by_folder["/proj/A"].file_count  == 1
    assert by_folder["/proj/B"].block_count == 1


def test_remove_folder(store):
    blocks = [
        _block("theorem", folder="/proj/A", index=1),
        _block("theorem", folder="/proj/B",
               source_file="/proj/B/main.tex", index=2),
    ]
    store.upsert(blocks, [_emb(b) for b in blocks])

    removed = store.remove_folder("/proj/A")
    assert removed == 1
    assert store._text_col.count() == 1
    assert store._math_col.count() == 1


def test_remove_folder_nonexistent(store):
    assert store.remove_folder("/no/such/folder") == 0


# ---------------------------------------------------------------------------
# _build_where
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("folder,env,expected", [
    (None,   None,        None),
    ("/foo", None,        {"folder":   {"$eq": "/foo"}}),
    (None,   "theorem",   {"env_type": {"$eq": "theorem"}}),
    ("/foo", "theorem",   {"$and": [
        {"folder":   {"$eq": "/foo"}},
        {"env_type": {"$eq": "theorem"}},
    ]}),
])
def test_build_where(folder, env, expected):
    assert _build_where(folder, env) == expected


# ---------------------------------------------------------------------------
# _should_boost_definitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("definition of compactness", True),
    ("Definition of foobar",      True),
    ("define limit",              True),
    ("find theorem about groups", False),
    ("search lemma 3",            False),
])
def test_should_boost_definitions(query, expected):
    assert _should_boost_definitions(query) == expected


# ---------------------------------------------------------------------------
# _merge_and_rank
# ---------------------------------------------------------------------------

def _raw_entry(env_type: str, scores: dict) -> dict:
    return {
        "doc":  "some text",
        "meta": {"env_type": env_type, "label": "", "section": "",
                 "source_file": "", "folder": ""},
        "scores": scores,
    }


def test_merge_ranks_by_score():
    raw = {
        "id1": _raw_entry("theorem",    {"text": 0.9}),
        "id2": _raw_entry("definition", {"text": 0.7}),
        "id3": _raw_entry("lemma",      {"text": 0.8}),
    }
    results = _merge_and_rank(raw, boost_definitions=False, n_results=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_merge_definition_boost():
    raw = {
        "id1": _raw_entry("theorem",    {"text": 0.9}),
        "id2": _raw_entry("definition", {"text": 0.85}),
    }
    results = _merge_and_rank(raw, boost_definitions=True, n_results=2)
    # definition (0.85 * 1.15 = 0.9775) should outrank theorem (0.9)
    assert results[0].env_type == "definition"


def test_merge_n_results_limit():
    raw = {f"id{i}": _raw_entry("theorem", {"text": i / 10}) for i in range(10)}
    results = _merge_and_rank(raw, boost_definitions=False, n_results=3)
    assert len(results) == 3


def test_merge_both_views_label():
    raw = {"id1": _raw_entry("lemma", {"text": 0.8, "math": 0.75})}
    results = _merge_and_rank(raw, boost_definitions=False, n_results=1)
    assert results[0].matched_view == "both"
    assert results[0].score == 0.8   # max of the two scores
