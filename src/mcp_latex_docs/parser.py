"""
parser.py - Extract structured blocks from LaTeX documents using pylatexenc.

Handles:
- theorem/lemma/definition/proof and related environments
- label extraction
- section hierarchy tracking
- input/include recursive resolution
- UTF-8, EUC-JP, Shift-JIS encoding detection
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chardet
from pylatexenc.latexwalker import (
    LatexEnvironmentNode,
    LatexMacroNode,
    LatexWalker,
    LatexNode,
)

# ---------------------------------------------------------------------------
# Supported environments
# ---------------------------------------------------------------------------

THEOREM_ENVS = {
    "theorem", "thm",
    "lemma", "lem",
    "proposition", "prop",
    "corollary", "cor",
    "definition", "defn", "dfn",
    "proof",
    "remark", "rem",
    "example", "ex",
    "conjecture", "conj",
    "claim",
    "notation",
    "observation",
}

ENV_NORMALIZE = {
    "thm":  "theorem",
    "lem":  "lemma",
    "prop": "proposition",
    "cor":  "corollary",
    "defn": "definition",
    "dfn":  "definition",
    "rem":  "remark",
    "ex":   "example",
    "conj": "conjecture",
}

SECTION_MACROS = {
    "chapter", "section", "subsection", "subsubsection",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Block:
    env_type:    str
    text:        str            # raw LaTeX of the environment body
    label:       str = ""       # from \label{}, empty if absent
    section:     str = ""       # enclosing section title, empty if unknown
    source_file: str = ""       # absolute path to the .tex file
    folder:      str = ""       # absolute path to the indexed folder
    block_id:    str = ""       # "<basename>::<env_type>::<index>"


# ---------------------------------------------------------------------------
# Encoding detection
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> str:
    raw = path.read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    # chardet sometimes returns 'ISO-8859-1' for ASCII — prefer UTF-8 in that case
    if encoding.upper() in ("ASCII", "ISO-8859-1"):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            pass
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _node_to_latex(node: LatexNode) -> str:
    """Return the original LaTeX source string for a node."""
    return node.latex_verbatim()


def _extract_label(node: LatexEnvironmentNode) -> str:
    """Extract the first \\label{...} found in an environment node."""
    if node.nodelist is None:
        return ""
    for child in node.nodelist:
        if isinstance(child, LatexMacroNode) and child.macroname == "label":
            if child.nodeargd and child.nodeargd.argnlist:
                for arg in child.nodeargd.argnlist:
                    if arg is not None:
                        return arg.latex_verbatim().strip("{} ")
    return ""


def _extract_section_title(node: LatexMacroNode) -> str:
    """Extract the title text from a section/subsection macro node."""
    if node.nodeargd and node.nodeargd.argnlist:
        for arg in node.nodeargd.argnlist:
            if arg is not None:
                return arg.latex_verbatim().strip("{} ")
    return ""


# ---------------------------------------------------------------------------
# Recursive \input / \include resolver
# ---------------------------------------------------------------------------

def _collect_tex_files(root: Path) -> list[Path]:
    """Collect .tex files in insertion order by resolving \\input/\\include."""
    visited: set[Path] = set()
    ordered: list[Path] = []

    def _visit(path: Path) -> None:
        path = path.resolve()
        if path in visited or not path.exists():
            return
        visited.add(path)
        ordered.append(path)

        try:
            source = _read_file(path)
        except Exception:
            return

        for match in re.finditer(
            r"\\(?:input|include)\{([^}]+)\}", source
        ):
            ref = match.group(1).strip()
            candidate = (path.parent / ref)
            if not candidate.suffix:
                candidate = candidate.with_suffix(".tex")
            _visit(candidate)

    # Start from all top-level .tex files in the folder
    root = root.resolve()
    for tex in sorted(root.rglob("*.tex")):
        _visit(tex)

    return ordered


# ---------------------------------------------------------------------------
# Per-file block extractor
# ---------------------------------------------------------------------------

def _parse_file(
    path: Path,
    folder: Path,
    block_counters: dict[str, int],
) -> list[Block]:
    try:
        source = _read_file(path)
    except Exception:
        return []

    try:
        walker = LatexWalker(source)
        nodelist, _, _ = walker.get_latex_nodes()
    except Exception:
        return []

    blocks: list[Block] = []
    current_section: str = ""
    basename = path.name

    def _visit_nodes(nodes) -> None:
        nonlocal current_section
        if nodes is None:
            return
        for node in nodes:
            if node is None:
                continue

            # Track section titles
            if isinstance(node, LatexMacroNode) and node.macroname in SECTION_MACROS:
                current_section = _extract_section_title(node)

            # Extract theorem-like environments
            elif isinstance(node, LatexEnvironmentNode):
                env = node.environmentname
                if env in THEOREM_ENVS:
                    normalized = ENV_NORMALIZE.get(env, env)
                    label = _extract_label(node)
                    body = _node_to_latex(node)

                    block_counters[normalized] = block_counters.get(normalized, 0) + 1
                    block_id = f"{basename}::{normalized}::{block_counters[normalized]}"

                    blocks.append(Block(
                        env_type=normalized,
                        text=body,
                        label=label,
                        section=current_section,
                        source_file=str(path),
                        folder=str(folder),
                        block_id=block_id,
                    ))
                else:
                    # Recurse into other environments (e.g. document, abstract)
                    _visit_nodes(node.nodelist)

    _visit_nodes(nodelist)
    return blocks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_folder(folder_path: str | Path) -> list[Block]:
    """
    Parse all LaTeX files in *folder_path* and return extracted blocks.

    Files are visited in \\input/\\include insertion order so that
    section-tracking remains coherent for multi-file documents.
    """
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    tex_files = _collect_tex_files(folder)
    block_counters: dict[str, int] = {}
    all_blocks: list[Block] = []

    for tex in tex_files:
        all_blocks.extend(_parse_file(tex, folder, block_counters))

    return all_blocks


def parse_file(file_path: str | Path, folder_path: str | Path | None = None) -> list[Block]:
    """Parse a single LaTeX file and return extracted blocks."""
    path = Path(file_path).resolve()
    folder = Path(folder_path).resolve() if folder_path else path.parent
    return _parse_file(path, folder, {})
