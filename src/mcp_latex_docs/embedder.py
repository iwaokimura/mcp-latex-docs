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
        """Embed a list of blocks with both models."""
        if not blocks:
            return []

        texts = [b.text for b in blocks]
        text_vecs = self._encode_e5_docs(texts)
        math_vecs = self._encode_mathberta(texts)

        return [
            BlockEmbedding(
                block_id=b.block_id,
                text_vec=text_vecs[i],
                math_vec=math_vecs[i],
            )
            for i, b in enumerate(blocks)
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
