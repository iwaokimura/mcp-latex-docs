"""
embedder.py - Embed LaTeX blocks and queries using two models.

Models:
  text  : intfloat/multilingual-e5-large-instruct  (English + Japanese prose)
  math  : witiko/mathberta                          (LaTeX math notation)

Every block is embedded with both models.
Queries are routed to one or both models based on content detection.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from mcp_latex_docs.parser import Block

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Query type detection
# ---------------------------------------------------------------------------

# Matches: $...$  \[...\]  \(...\)  \command  or standalone LaTeX backslash macros
_LATEX_RE = re.compile(
    r"\$\$[^$]+\$\$"       # display math $$...$$  (must come before inline)
    r"|\$[^$]+\$"          # inline math $...$
    r"|\\[\(\[\]\)]"       # \(  \[  \]  \)
    r"|\\[a-zA-Z]+[{\s]"   # \command{ or \command<space>
    r"|\\[a-zA-Z]+"        # bare \command
)

# Prose word: 3+ Unicode letters (covers English and Japanese;
# filters out 2-char math tokens like dx, dt, dy)
_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


class QueryType(Enum):
    TEXT  = auto()
    MATH  = auto()
    MIXED = auto()


def detect_query_type(query: str) -> QueryType:
    """Classify a query as plain text, LaTeX math, or mixed."""
    if not _LATEX_RE.search(query):
        return QueryType.TEXT

    # Strip LaTeX spans and math-adjacent noise (_  ^  digits  single chars)
    remaining = _LATEX_RE.sub("", query)
    remaining = re.sub(r"[_^{}\[\]()|,.\d]", " ", remaining)

    # Check whether any prose words survive
    words = _WORD_RE.findall(remaining)
    if words:
        return QueryType.MIXED
    return QueryType.MATH


# ---------------------------------------------------------------------------
# mathberta mean-pool helper
# ---------------------------------------------------------------------------

def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


# ---------------------------------------------------------------------------
# Embedding result
# ---------------------------------------------------------------------------

@dataclass
class BlockEmbedding:
    block_id:   str
    text_vec:   list[float]   # from multilingual-e5
    math_vec:   list[float]   # from mathberta
    text:       str = ""      # sub-block text; empty string means use original block text


@dataclass
class QueryEmbedding:
    query_type: QueryType
    text_vec:   list[float] | None   # None when query is MATH-only
    math_vec:   list[float] | None   # None when query is TEXT-only


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

# Instruction prefix for multilingual-e5-large-instruct query encoding
_E5_TASK = (
    "Retrieve relevant mathematical theorems, definitions, lemmas, or proofs"
)

# mathberta's nominal context window; leave a small margin for CLS / SEP tokens.
_MB_MAX_TOKENS = 500


# ---------------------------------------------------------------------------
# Block-splitting helpers
# ---------------------------------------------------------------------------

def _greedy_pack_tokens(
    tok: AutoTokenizer,
    parts: list[str],
    sep: str,
    max_tokens: int,
) -> list[str]:
    """Greedily join *parts* (separated by *sep*) into chunks ≤ *max_tokens*."""
    chunks: list[str] = []
    current: list[str] = []

    for part in parts:
        candidate = sep.join(current + [part]) if current else part
        if _count_mb_tokens(tok, candidate) <= max_tokens:
            current.append(part)
        else:
            if current:
                chunks.append(sep.join(current))
            current = [part]

    if current:
        chunks.append(sep.join(current))

    return chunks or [sep.join(parts)]


def _count_mb_tokens(tok: AutoTokenizer, text: str) -> int:
    """Count mathberta tokens without triggering the long-sequence warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return len(tok.encode(text, add_special_tokens=True))


def _split_text_for_mb(
    tok: AutoTokenizer,
    text: str,
    max_tokens: int = _MB_MAX_TOKENS,
) -> list[str]:
    """Split *text* into chunks ≤ *max_tokens* mathberta tokens.

    Tries natural boundaries in order: paragraphs (\\n\\n), sentence-ending
    punctuation, then arbitrary whitespace.  Falls back to a hard split by
    token IDs if no boundary produces small-enough pieces.
    """
    if _count_mb_tokens(tok, text) <= max_tokens:
        return [text]

    for pattern, sep in [
        (r"\n{2,}", "\n\n"),
        (r"(?<=[.?!])\s+", " "),
        (r"\s+", " "),
    ]:
        parts = re.split(pattern, text.strip())
        if len(parts) > 1:
            packed = _greedy_pack_tokens(tok, parts, sep, max_tokens)
            result: list[str] = []
            for chunk in packed:
                result.extend(_split_text_for_mb(tok, chunk, max_tokens))
            return result

    # Absolute fallback: hard-split by token IDs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ids = tok.encode(text, add_special_tokens=False)
    mid = len(ids) // 2
    return [
        tok.decode(ids[:mid]),
        tok.decode(ids[mid:]),
    ]


class Embedder:
    """
    Wraps both embedding models and provides embed methods for blocks and queries.

    Models are loaded lazily on first use to keep import time fast.
    """

    def __init__(self) -> None:
        self._device = _get_device()
        self._e5: SentenceTransformer | None = None
        self._mb_model: AutoModel | None = None
        self._mb_tok: AutoTokenizer | None = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_e5(self) -> SentenceTransformer:
        if self._e5 is None:
            self._e5 = SentenceTransformer(
                "intfloat/multilingual-e5-large-instruct",
                device=str(self._device),
            )
        return self._e5

    def _load_mathberta(self) -> tuple[AutoTokenizer, AutoModel]:
        if self._mb_model is None:
            self._mb_tok = AutoTokenizer.from_pretrained("witiko/mathberta")
            self._mb_model = AutoModel.from_pretrained("witiko/mathberta").to(
                self._device
            )
            self._mb_model.eval()
        return self._mb_tok, self._mb_model

    # ------------------------------------------------------------------
    # Low-level encode helpers
    # ------------------------------------------------------------------

    def _encode_e5_docs(self, texts: list[str]) -> list[list[float]]:
        model = self._load_e5()
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.tolist()

    def _encode_e5_query(self, query: str) -> list[float]:
        model = self._load_e5()
        vec = model.encode(
            query,
            prompt=f"Instruct: {_E5_TASK}\nQuery: ",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    def _encode_mathberta(self, texts: list[str]) -> list[list[float]]:
        tok, model = self._load_mathberta()
        encoded = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            out = model(**encoded)
        pooled = _mean_pool(out.last_hidden_state, encoded["attention_mask"])
        normed = F.normalize(pooled, p=2, dim=1)
        return normed.cpu().tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_blocks(self, blocks: Sequence[Block]) -> list[BlockEmbedding]:
        """Embed a list of blocks with both models.

        Blocks whose text exceeds mathberta's 512-token limit are split at
        natural boundaries (paragraphs, then sentences).  Each part receives
        its own ``BlockEmbedding`` with an ID of the form
        ``<block_id>::part<N>``.
        """
        if not blocks:
            return []

        tok, _ = self._load_mathberta()

        # Expand each block into (embedding_id, text) pairs, splitting if needed.
        pairs: list[tuple[str, str]] = []   # (block_id, text)
        for b in blocks:
            parts = _split_text_for_mb(tok, b.text)
            if len(parts) == 1:
                pairs.append((b.block_id, b.text))
            else:
                pairs.extend(
                    (f"{b.block_id}::part{i}", chunk)
                    for i, chunk in enumerate(parts)
                )

        all_texts = [t for _, t in pairs]
        text_vecs = self._encode_e5_docs(all_texts)
        math_vecs = self._encode_mathberta(all_texts)

        return [
            BlockEmbedding(
                block_id=bid,
                text_vec=text_vecs[i],
                math_vec=math_vecs[i],
                text=txt,
            )
            for i, (bid, txt) in enumerate(pairs)
        ]

    def embed_query(self, query: str) -> QueryEmbedding:
        """
        Embed a query, routing to the appropriate model(s).

        - TEXT  query  → e5 only
        - MATH  query  → mathberta only
        - MIXED query  → both
        """
        qtype = detect_query_type(query)

        text_vec: list[float] | None = None
        math_vec: list[float] | None = None

        if qtype in (QueryType.TEXT, QueryType.MIXED):
            text_vec = self._encode_e5_query(query)

        if qtype in (QueryType.MATH, QueryType.MIXED):
            math_vec = self._encode_mathberta([query])[0]

        return QueryEmbedding(query_type=qtype, text_vec=text_vec, math_vec=math_vec)
