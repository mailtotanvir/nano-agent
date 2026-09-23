"""Canonical Python representation of the code-speed patch contract.

The Rust ``core/patch`` crate remains authoritative at runtime. These helpers
mirror its parser so corpus construction and diagnostic replay use the same
SEARCH/REPLACE representation, including legacy unified-diff ingestion.
"""
from __future__ import annotations

from pathlib import Path

MARK_SEARCH = "<<<<<<< SEARCH"
MARK_DIVIDER = "======="
MARK_REPLACE = ">>>>>>> REPLACE"
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompt" / "system.txt"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").rstrip()


def parse_blocks(text: str, default_file: str | None):
    """Return ``(file, search, replace)`` blocks, or ``None`` if malformed."""
    blocks = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].rstrip() == MARK_SEARCH:
            file = default_file
            if i > 0:
                previous = lines[i - 1].strip()
                if previous.startswith("file:"):
                    hint = previous[len("file:"):].strip()
                    if hint:
                        file = hint
            if file is None:
                return None
            j = i + 1
            search = []
            while j < len(lines) and lines[j].rstrip() != MARK_DIVIDER:
                if lines[j].rstrip() in (MARK_SEARCH, MARK_REPLACE):
                    return None
                search.append(lines[j])
                j += 1
            if j >= len(lines):
                return None
            j += 1
            replace = []
            while j < len(lines) and lines[j].rstrip() != MARK_REPLACE:
                if lines[j].rstrip() in (MARK_SEARCH, MARK_DIVIDER):
                    return None
                replace.append(lines[j])
                j += 1
            if j >= len(lines):
                return None
            blocks.append((file, "\n".join(search), "\n".join(replace)))
            i = j + 1
        else:
            i += 1
    return blocks or parse_unified_diff(text, default_file)


def parse_unified_diff(text: str, default_file: str | None):
    """Mirror ``core/patch``'s compatibility parser for unified diffs."""
    lines = text.splitlines()
    blocks = []
    file = default_file
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("+++ "):
            name = line[4:].strip()
            name = name.removeprefix("b/")
            if name and name != "/dev/null":
                file = name
            i += 1
            continue
        if line.startswith(("--- ", "diff ", "index ")):
            i += 1
            continue
        if not line.startswith("@@"):
            i += 1
            continue
        if file is None:
            return None
        search = []
        replace = []
        i += 1
        while i < len(lines):
            hunk = lines[i]
            if hunk.startswith(("@@", "--- ", "+++ ", "diff ")):
                break
            if hunk.startswith("\\"):
                i += 1
                continue
            if hunk.startswith("+"):
                replace.append(hunk[1:])
            elif hunk.startswith("-"):
                search.append(hunk[1:])
            elif hunk.startswith(" "):
                search.append(hunk[1:])
                replace.append(hunk[1:])
            elif not hunk:
                search.append("")
                replace.append("")
            else:
                break
            i += 1
        blocks.append((file, "\n".join(search), "\n".join(replace)))
    return blocks


def _normalize(text: str) -> str:
    return " ".join(text.split())


def apply_block(content: str, search: str, replace: str):
    """Apply one block with the production exact/fuzzy/ambiguity behavior."""
    if search == replace:
        return None
    positions = []
    start = 0
    while search:
        position = content.find(search, start)
        if position < 0:
            break
        positions.append(position)
        start = position + 1
    if len(positions) == 1:
        position = positions[0]
        return content[:position] + replace + content[position + len(search):]
    if len(positions) > 1:
        return None

    content_lines = content.splitlines(keepends=True)
    search_norm = [_normalize(line) for line in search.split("\n")]
    if not search_norm:
        return None
    width = len(search_norm)
    plain = [_normalize(line) for line in content_lines]
    hit = None
    for index in range(len(plain) - width + 1):
        if plain[index:index + width] == search_norm:
            if hit is not None:
                return None
            hit = index
    if hit is None:
        return None
    output = "".join(content_lines[:hit]) + replace
    if content_lines[hit + width - 1].endswith("\n") and not replace.endswith("\n"):
        output += "\n"
    return output + "".join(content_lines[hit + width:])


def reconstruct_candidate(source: str, patch: str | None):
    blocks = parse_blocks(patch or "", "candidate.py")
    if not blocks:
        return None, blocks is not None, False
    candidate = source
    for _, search, replace in blocks:
        candidate = apply_block(candidate, search, replace)
        if candidate is None:
            return None, True, False
    return candidate, True, True


def canonicalize_patch(patch: str | None, default_file: str = "candidate.py") -> str:
    """Convert an accepted legacy patch into canonical SEARCH/REPLACE blocks."""
    blocks = parse_blocks(patch or "", default_file)
    if not blocks:
        raise ValueError("accepted proposal has no parseable patch blocks")
    rendered = []
    for file, search, replace in blocks:
        rendered.extend([
            f"file: {file}", MARK_SEARCH, search, MARK_DIVIDER, replace, MARK_REPLACE,
        ])
    return "\n".join(rendered) + "\n"
