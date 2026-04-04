"""Tests for embedder.py — query detection tested directly; encode paths mocked."""

from unittest.mock import MagicMock, patch

import pytest

from mcp_latex_docs.embedder import (
    Embedder,
    QueryType,
    _greedy_pack_tokens,
    _split_text_for_mb,
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

    # Mock the mathberta tokenizer to report short sequences (≤ 500 tokens)
    # so that embed_blocks never triggers splitting in unit tests.
    mock_tok = MagicMock()
    mock_tok.encode.return_value = list(range(10))
    mock_model = MagicMock()
    emb._load_mathberta = MagicMock(return_value=(mock_tok, mock_model))
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


# ---------------------------------------------------------------------------
# _split_text_for_mb / _greedy_pack_tokens
# ---------------------------------------------------------------------------

def _tok_by_length(text: str, add_special_tokens: bool = True) -> list[int]:
    """Fake tokenizer: 1 token per character (plus 2 for CLS/SEP when requested)."""
    n = len(text) + (2 if add_special_tokens else 0)
    return list(range(n))


def _make_mock_tok():
    tok = MagicMock()
    tok.encode.side_effect = _tok_by_length
    # decode: just join the "token ids" as hex strings (not used in assertions)
    tok.decode.side_effect = lambda ids, **kw: f"<decoded:{len(ids)}>"
    return tok


def test_split_text_noop_when_short():
    tok = _make_mock_tok()
    # 10 chars + 2 special = 12 tokens, well under 500
    result = _split_text_for_mb(tok, "short text", max_tokens=500)
    assert result == ["short text"]


def test_split_text_on_paragraphs():
    # Each paragraph is 5 chars + 2 = 7 tokens; together "aaaaa\n\nbbbbb" is
    # 13 chars + 2 = 15 tokens.  With max_tokens=10 the combined text exceeds
    # the limit but each individual paragraph fits.
    tok = _make_mock_tok()
    result = _split_text_for_mb(tok, "aaaaa\n\nbbbbb", max_tokens=10)
    assert result == ["aaaaa", "bbbbb"]


def test_split_text_on_sentences():
    # No paragraph break, but sentence ends with ". "
    tok = _make_mock_tok()
    # "aaa. bbb" = 8 chars + 2 = 10 tokens → split at ". "
    # "aaa." = 4 + 2 = 6 tokens (fits), "bbb" = 3 + 2 = 5 tokens (fits)
    result = _split_text_for_mb(tok, "aaa. bbb", max_tokens=9)
    assert result == ["aaa.", "bbb"]


def test_greedy_pack_tokens_combines_short_parts():
    tok = _make_mock_tok()
    # "a" + "b" joined as "a\n\nb" = 5 chars + 2 = 7 tokens ≤ max 10
    result = _greedy_pack_tokens(tok, ["a", "b"], "\n\n", max_tokens=10)
    assert result == ["a\n\nb"]


def test_greedy_pack_tokens_flushes_when_full():
    tok = _make_mock_tok()
    # Each "aaaa" alone = 6 tokens; two together "aaaa\n\naaaa" = 10 + 2 = 12 > 8
    result = _greedy_pack_tokens(tok, ["aaaa", "aaaa"], "\n\n", max_tokens=8)
    assert result == ["aaaa", "aaaa"]


# ---------------------------------------------------------------------------
# embed_blocks — splitting behaviour
# ---------------------------------------------------------------------------

def test_embed_blocks_splits_large_block():
    """A block over the token limit is split into multiple BlockEmbeddings."""
    emb = Embedder()

    # Tokenizer: text with "\n\n" reports 600 tokens; half-texts report 200.
    mock_tok = MagicMock()
    def _encode(text, add_special_tokens=True):
        return list(range(600)) if "\n\n" in text else list(range(200))
    mock_tok.encode.side_effect = _encode
    mock_model = MagicMock()
    emb._load_mathberta = MagicMock(return_value=(mock_tok, mock_model))

    emb._encode_e5_docs   = MagicMock(return_value=[[0.1] * 1024, [0.1] * 1024])
    emb._encode_mathberta = MagicMock(return_value=[[0.2] * 768,  [0.2] * 768])

    block = Block(
        env_type="proof",
        text="First paragraph.\n\nSecond paragraph.",
        block_id="file.tex::proof::0",
    )
    results = emb.embed_blocks([block])

    assert len(results) == 2
    assert results[0].block_id == "file.tex::proof::0::part0"
    assert results[1].block_id == "file.tex::proof::0::part1"
    assert results[0].text == "First paragraph."
    assert results[1].text == "Second paragraph."
