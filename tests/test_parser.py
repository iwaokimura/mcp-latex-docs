"""Basic smoke tests for parser.py."""

import textwrap
from pathlib import Path

import pytest

from mcp_latex_docs.parser import parse_file, Block


SAMPLE_TEX = textwrap.dedent(r"""
    \documentclass{article}
    \usepackage{amsthm}
    \begin{document}

    \section{Introduction}

    \begin{definition}\label{def:limit}
    Let $f : \mathbb{R} \to \mathbb{R}$. We say $\lim_{x \to a} f(x) = L$ if
    for every $\varepsilon > 0$ there exists $\delta > 0$ such that
    $|f(x) - L| < \varepsilon$ whenever $0 < |x - a| < \delta$.
    \end{definition}

    \begin{theorem}\label{thm:squeeze}
    If $f(x) \le g(x) \le h(x)$ and $\lim f = \lim h = L$, then $\lim g = L$.
    \end{theorem}

    \begin{proof}
    Follows directly from the definition of limit.
    \end{proof}

    \begin{lemma}
    Every continuous function on a closed interval is bounded.
    \end{lemma}

    \end{document}
""")


@pytest.fixture
def sample_tex(tmp_path):
    f = tmp_path / "sample.tex"
    f.write_text(SAMPLE_TEX, encoding="utf-8")
    return f


def test_block_count(sample_tex):
    blocks = parse_file(sample_tex)
    assert len(blocks) == 4


def test_env_types(sample_tex):
    blocks = parse_file(sample_tex)
    types = [b.env_type for b in blocks]
    assert types == ["definition", "theorem", "proof", "lemma"]


def test_labels(sample_tex):
    blocks = parse_file(sample_tex)
    assert blocks[0].label == "def:limit"
    assert blocks[1].label == "thm:squeeze"
    assert blocks[2].label == ""


def test_section(sample_tex):
    blocks = parse_file(sample_tex)
    for block in blocks:
        assert block.section == "Introduction"


def test_block_id(sample_tex):
    blocks = parse_file(sample_tex)
    assert blocks[0].block_id == "sample.tex::definition::1"
    assert blocks[1].block_id == "sample.tex::theorem::1"


def test_text_contains_math(sample_tex):
    blocks = parse_file(sample_tex)
    assert r"\mathbb{R}" in blocks[0].text
