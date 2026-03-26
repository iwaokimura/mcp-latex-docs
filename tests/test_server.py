"""Tests for server.py tool dispatch — store and embedder are mocked."""

from unittest.mock import MagicMock, patch

import pytest

import mcp_latex_docs.server as srv
from mcp_latex_docs.embedder import QueryEmbedding, QueryType
from mcp_latex_docs.parser import Block
from mcp_latex_docs.store import FolderInfo, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(env_type: str = "theorem", label: str = "") -> Block:
    return Block(
        env_type=env_type,
        text=r"Let $f$ be continuous.",
        label=label,
        section="Section 1",
        source_file="/foo/main.tex",
        folder="/foo",
        block_id=f"main.tex::{env_type}::1",
    )


def _make_result(env_type: str = "theorem", score: float = 0.9) -> SearchResult:
    return SearchResult(
        block_id=f"main.tex::{env_type}::1",
        env_type=env_type,
        label="",
        section="Introduction",
        source_file="/foo/main.tex",
        folder="/foo",
        text=r"Let $f$ be continuous.",
        score=score,
        matched_view="text",
    )


def _mock_embedder(query_type: QueryType = QueryType.TEXT) -> MagicMock:
    emb = MagicMock()
    emb.embed_blocks.return_value = []
    emb.embed_query.return_value = QueryEmbedding(
        query_type=query_type,
        text_vec=[0.1] * 8,
        math_vec=None,
    )
    return emb


def _mock_store() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# index_folder
# ---------------------------------------------------------------------------

def test_index_folder_success():
    blocks = [_make_block("theorem"), _make_block("definition")]
    with (
        patch("mcp_latex_docs.server.parse_folder", return_value=blocks),
        patch("mcp_latex_docs.server._get_embedder", return_value=_mock_embedder()),
        patch("mcp_latex_docs.server._get_store", return_value=_mock_store()),
    ):
        result = srv._tool_index_folder("/foo")
    assert "2 block(s)" in result
    assert "/foo" in result


def test_index_folder_no_blocks():
    with patch("mcp_latex_docs.server.parse_folder", return_value=[]):
        result = srv._tool_index_folder("/empty")
    assert "No theorem-like blocks" in result


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_returns_formatted_results():
    mock_store = _mock_store()
    mock_store.search.return_value = [_make_result("theorem", 0.95)]

    with (
        patch("mcp_latex_docs.server._get_embedder", return_value=_mock_embedder()),
        patch("mcp_latex_docs.server._get_store",    return_value=mock_store),
    ):
        result = srv._tool_search("find theorem", None, None, 5)

    assert "THEOREM" in result
    assert "0.95" in result
    assert r"Let $f$ be continuous." in result


def test_search_no_results():
    mock_store = _mock_store()
    mock_store.search.return_value = []

    with (
        patch("mcp_latex_docs.server._get_embedder", return_value=_mock_embedder()),
        patch("mcp_latex_docs.server._get_store",    return_value=mock_store),
    ):
        result = srv._tool_search("nothing", None, None, 5)

    assert "No results" in result


def test_search_passes_filters():
    mock_store = _mock_store()
    mock_store.search.return_value = []

    with (
        patch("mcp_latex_docs.server._get_embedder", return_value=_mock_embedder()),
        patch("mcp_latex_docs.server._get_store",    return_value=mock_store),
    ):
        srv._tool_search("compactness", "/foo", "definition", 3)

    call_kwargs = mock_store.search.call_args.kwargs
    assert call_kwargs["folder_path"] == "/foo"
    assert call_kwargs["env_type"]    == "definition"
    assert call_kwargs["n_results"]   == 3


# ---------------------------------------------------------------------------
# get_by_label
# ---------------------------------------------------------------------------

def test_get_by_label_found():
    r = _make_result("theorem")
    r.label = "thm:main"
    mock_store = _mock_store()
    mock_store.get_by_label.return_value = r

    with patch("mcp_latex_docs.server._get_store", return_value=mock_store):
        result = srv._tool_get_by_label("thm:main", None)

    assert "THEOREM" in result
    assert "thm:main" in result


def test_get_by_label_not_found():
    mock_store = _mock_store()
    mock_store.get_by_label.return_value = None

    with patch("mcp_latex_docs.server._get_store", return_value=mock_store):
        result = srv._tool_get_by_label("missing", None)

    assert "No block" in result
    assert "missing" in result


# ---------------------------------------------------------------------------
# list_folders
# ---------------------------------------------------------------------------

def test_list_folders_with_data():
    mock_store = _mock_store()
    mock_store.list_folders.return_value = [
        FolderInfo("/foo", file_count=2, block_count=15),
        FolderInfo("/bar", file_count=1, block_count=7),
    ]

    with patch("mcp_latex_docs.server._get_store", return_value=mock_store):
        result = srv._tool_list_folders()

    assert "/foo" in result
    assert "15 block(s)" in result
    assert "/bar" in result


def test_list_folders_empty():
    mock_store = _mock_store()
    mock_store.list_folders.return_value = []

    with patch("mcp_latex_docs.server._get_store", return_value=mock_store):
        result = srv._tool_list_folders()

    assert "No folders" in result


# ---------------------------------------------------------------------------
# remove_folder
# ---------------------------------------------------------------------------

def test_remove_folder_success():
    mock_store = _mock_store()
    mock_store.remove_folder.return_value = 12

    with patch("mcp_latex_docs.server._get_store", return_value=mock_store):
        result = srv._tool_remove_folder("/foo")

    assert "12 block(s)" in result
    assert "/foo" in result


def test_remove_folder_not_found():
    mock_store = _mock_store()
    mock_store.remove_folder.return_value = 0

    with patch("mcp_latex_docs.server._get_store", return_value=mock_store):
        result = srv._tool_remove_folder("/no/such")

    assert "No indexed blocks" in result
