"""Tests for embedder.py — query detection tested directly; encode paths mocked."""

from unittest.mock import MagicMock, patch

import pytest

from mcp_latex_docs.embedder import (
    Embedder,
    QueryType,
    detect_query_type,
)
from mcp_latex_docs.parser import Block


# ---------------------------------------------------------------------------
# detect_query_type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query, expected", [
    # plain text
    ("search the definition of compactness", QueryType.TEXT),
    ("find all lemmas about continuity", QueryType.TEXT),
    # Japanese prose
    ("コンパクト性の定義を探す", QueryType.TEXT),
    # pure math
    (r"$\phi_{foobar}$", QueryType.MATH),
    (r"\int_0^1 f(x)\,dx", QueryType.MATH),
    # mixed
    (r"definition of $\phi_{foobar}$", QueryType.MIXED),
    (r"search the definition of the symbol $\phi_{foobar}$", QueryType.MIXED),
    (r"theorems about \lim_{x \to 0} f(x)", QueryType.MIXED),
])
def test_detect_query_type(query, expected):
    assert detect_query_type(query) == expected


# ---------------------------------------------------------------------------
# Embedder.embed_query — routing
# ---------------------------------------------------------------------------

def _make_embedder_with_mocks():
    """Return an Embedder whose internal encode methods are mocked."""
    emb = Embedder()
    emb._encode_e5_query     = MagicMock(return_value=[0.1] * 1024)
    emb._encode_mathberta    = MagicMock(return_value=[[0.2] * 768])
    emb._encode_e5_docs      = MagicMock(return_value=[[0.1] * 1024])
    return emb


def test_text_query_uses_e5_only():
    emb = _make_embedder_with_mocks()
    result = emb.embed_query("find definition of compactness")
    assert result.query_type == QueryType.TEXT
    assert result.text_vec is not None
    assert result.math_vec is None
    emb._encode_e5_query.assert_called_once()
    emb._encode_mathberta.assert_not_called()


def test_math_query_uses_mathberta_only():
    emb = _make_embedder_with_mocks()
    result = emb.embed_query(r"$\phi_{foobar}$")
    assert result.query_type == QueryType.MATH
    assert result.text_vec is None
    assert result.math_vec is not None
    emb._encode_e5_query.assert_not_called()
    emb._encode_mathberta.assert_called_once()


def test_mixed_query_uses_both():
    emb = _make_embedder_with_mocks()
    result = emb.embed_query(r"definition of $\phi_{foobar}$")
    assert result.query_type == QueryType.MIXED
    assert result.text_vec is not None
    assert result.math_vec is not None
    emb._encode_e5_query.assert_called_once()
    emb._encode_mathberta.assert_called_once()


# ---------------------------------------------------------------------------
# Embedder.embed_blocks
# ---------------------------------------------------------------------------

def _make_block(i: int) -> Block:
    return Block(
        env_type="theorem",
        text=rf"Let $f_{i}$ be continuous.",
        block_id=f"file.tex::theorem::{i}",
    )


def test_embed_blocks_returns_correct_ids():
    emb = _make_embedder_with_mocks()
    emb._encode_e5_docs   = MagicMock(return_value=[[0.1] * 1024, [0.1] * 1024])
    emb._encode_mathberta = MagicMock(return_value=[[0.2] * 768,  [0.2] * 768])

    blocks = [_make_block(1), _make_block(2)]
    results = emb.embed_blocks(blocks)

    assert len(results) == 2
    assert results[0].block_id == "file.tex::theorem::1"
    assert results[1].block_id == "file.tex::theorem::2"
    assert len(results[0].text_vec) == 1024
    assert len(results[0].math_vec) == 768


def test_embed_blocks_empty():
    emb = _make_embedder_with_mocks()
    assert emb.embed_blocks([]) == []
