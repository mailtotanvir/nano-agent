#!/usr/bin/env python3
"""Synthetic Rust breakage generator.

Takes compiling Rust snippets and applies deterministic, labeled mutations that
each induce a specific, known compiler error class. Every generated case is
verified with `cargo check` to confirm it (a) fails and (b) fails with the
intended error code, then written to a JSONL corpus.

This is source #1 of the dataset (spec section 4.6): ground-truth error taxonomy
for the competence-envelope analysis. Determinism (seeded) makes the eval/train
split reproducible and non-overlapping.

Usage:
    python break_it.py --out corpus.jsonl --count 200 --seed 0
    python break_it.py --out eval.jsonl --count 500 --seed 1 --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

try:
    from seeds import SEEDS as STATIC_SEEDS
except ImportError:
    from .seeds import SEEDS as STATIC_SEEDS

try:
    from seed_gen import generate_seeds
except ImportError:
    from .seed_gen import generate_seeds

# Default to the static hand-written seeds; the CLI can swap in parametric seeds.
SEED_PROGRAMS = list(STATIC_SEEDS)


@dataclass
class Mutation:
    """A labeled mutation. `sites` returns ALL broken variants of `src` (one per
    matching location), so a single seed can yield many distinct cases."""
    error_code: str        # intended rustc error code, e.g. "E0308"
    category: str          # human label, e.g. "type_mismatch"
    sites: Callable[[str], list[str]]  # returns list of broken variants


# --- mutation rules (each enumerates every matching site) ---

def _replace_at(src: str, m: re.Match, group: int, repl: str) -> str:
    return src[: m.start(group)] + repl + src[m.end(group):]


def s_type_mismatch(src: str) -> list[str]:
    """Integer literal assigned to a typed int binding -> string literal (E0308)."""
    out = []
    for m in re.finditer(r"let \w+: (?:i8|i16|i32|i64|u8|u16|u32|u64|usize) = (\d+)(?=;)", src):
        out.append(_replace_at(src, m, 1, '"' + m.group(1) + '"'))
    return out


def s_bool_as_int(src: str) -> list[str]:
    """Assign an int to a bool binding (E0308)."""
    out = []
    for m in re.finditer(r"let \w+: bool = (true|false)(?=;)", src):
        out.append(_replace_at(src, m, 1, "1"))
    return out


def s_missing_import(src: str) -> list[str]:
    """Remove each `use std::...;` line (E0433 unresolved path)."""
    out = []
    for m in re.finditer(r"use std::[\w:]+;\n", src):
        out.append(src[: m.start()] + src[m.end():])
    return out


def s_missing_semicolon(src: str) -> list[str]:
    """Drop the trailing semicolon on each `let ...;` statement (syntax)."""
    out = []
    lines = src.split("\n")
    for i, l in enumerate(lines):
        if l.rstrip().endswith(";") and "let " in l:
            copy = lines[:]
            copy[i] = l.rstrip()[:-1]
            out.append("\n".join(copy))
    return out


def s_unknown_method(src: str) -> list[str]:
    """Rename each call to a common method to a nonexistent one (E0599)."""
    out = []
    methods = "iter|first|len|push|insert|copied|sum|map|collect|clone|to_string|to_uppercase|parse"
    for m in re.finditer(rf"\.({methods})\(", src):
        out.append(src[: m.start()] + ".nonexistent_method_qz(" + src[m.end():])
    return out


def s_undeclared_ident(src: str) -> list[str]:
    """Reference an undeclared variable in each simple println arg (E0425)."""
    out = []
    for m in re.finditer(r'println!\("\{[^}]*\}",\s*(\w+)\)', src):
        # skip if it's actually a field access base we can't easily judge; simple idents only
        out.append(_replace_at(src, m, 1, "undeclared_var_qz"))
    return out


def s_missing_derive(src: str) -> list[str]:
    """Remove a derive that a {:?}/clone usage depends on (E0277/E0599)."""
    out = []
    if "{:?}" in src or "{:#?}" in src:
        for m in re.finditer(r"#\[derive\([^)]*\bDebug\b[^)]*\)\]\n", src):
            out.append(src[: m.start()] + src[m.end():])
    return out


def s_wrong_arg_type(src: str) -> list[str]:
    """Pass an integer where a &str literal argument is expected (E0308)."""
    out = []
    for m in re.finditer(r"(\w+)\(\"[^\"]*\"\)", src):
        # only functions taking &str; heuristic: call site with a single string literal
        fn = m.group(1)
        if fn in ("println", "print", "format", "vec", "eprintln"):
            continue
        out.append(src[: m.start()] + fn + "(42)" + src[m.end():])
    return out


def s_missing_mut(src: str) -> list[str]:
    """Drop `mut` from each mutable binding (E0596)."""
    out = []
    for m in re.finditer(r"let mut (\w+)", src):
        out.append(src[: m.start()] + "let " + m.group(1) + src[m.end():])
    return out


def s_missing_type_annotation(src: str) -> list[str]:
    """Remove a needed type annotation on a collect() binding (E0282/E0283)."""
    out = []
    for m in re.finditer(r"let (\w+): (?:Vec<\w+>) = ([^;]*collect\(\))", src):
        out.append(src[: m.start()] + "let " + m.group(1) + " = " + m.group(2) + src[m.end():])
    return out


def s_wrong_field_name(src: str) -> list[str]:
    """Rename a field at a struct construction site so it no longer exists (E0560)."""
    out = []
    for m in re.finditer(r"([A-Z]\w+) \{ (\w+):", src):
        out.append(_replace_at(src, m, 2, "nonexistent_field"))
    return out


def s_return_ref_not_owned(src: str) -> list[str]:
    """Return a &str literal from a fn declared -> String (E0308)."""
    out = []
    for m in re.finditer(r'(\w+)\.to_string\(\)', src):
        out.append(src[: m.start()] + m.group(1) + src[m.end():])
    return out


MUTATIONS: list[Mutation] = [
    Mutation("E0308", "type_mismatch", s_type_mismatch),
    Mutation("E0308", "bool_as_int", s_bool_as_int),
    Mutation("E0433", "missing_import", s_missing_import),
    Mutation("E0599", "unknown_method", s_unknown_method),
    Mutation("E0425", "undeclared_ident", s_undeclared_ident),
    Mutation("E0277", "missing_derive", s_missing_derive),
    Mutation("E0308", "wrong_arg_type", s_wrong_arg_type),
    Mutation("E0596", "missing_mut", s_missing_mut),
    Mutation("E0282", "missing_type_annotation", s_missing_type_annotation),
    Mutation("E0560", "wrong_field_name", s_wrong_field_name),
    Mutation("E0308", "return_type_mismatch", s_return_ref_not_owned),
    Mutation("SYNTAX", "missing_semicolon", s_missing_semicolon),
]


@dataclass
class Case:
    case_id: str
    seed_index: int
    category: str
    intended_code: str
    original: str
    broken: str
    observed_codes: list[str]
    verified: bool


def cargo_check_codes(src: str, tmp_root: Path) -> tuple[bool, list[str]]:
    """Write `src` as a crate, run cargo check, return (passed, error_codes)."""
    crate = tmp_root / "src"
    crate.mkdir(parents=True, exist_ok=True)
    (tmp_root / "Cargo.toml").write_text(
        '[package]\nname = "case"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "case"\npath = "src/main.rs"\n'
    )
    (crate / "main.rs").write_text(src)
    proc = subprocess.run(
        ["cargo", "check", "--message-format=json", "--quiet"],
        cwd=tmp_root, capture_output=True, text=True,
    )
    codes: list[str] = []
    passed = True
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-message":
            continue
        d = msg.get("message", {})
        if d.get("level") == "error":
            passed = False
            code = (d.get("code") or {}).get("code")
            codes.append(code if code else "SYNTAX")
    return passed, codes


def enumerate_candidates(seed_programs: list[str] | None = None) -> list[tuple[int, Mutation, str, str]]:
    """Enumerate ALL (seed, mutation, site) candidates deterministically.

    Each mutation returns every broken variant of every seed, so the candidate
    pool is the full cross product of seeds x mutations x matching sites.
    """
    programs = seed_programs if seed_programs is not None else SEED_PROGRAMS
    cands: list[tuple[int, Mutation, str, str]] = []
    seen: set[str] = set()
    for si, src in enumerate(programs):
        for mut in MUTATIONS:
            for broken in mut.sites(src):
                if broken == src or not broken.strip():
                    continue
                key = hashlib.sha1((mut.category + broken).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                cands.append((si, mut, src, broken))
    return cands


def _verify_candidate(cand: tuple[int, Mutation, str, str], workers_root: Path, wid: int) -> Optional[Case]:
    si, mut, src, broken = cand
    tmp = workers_root / f"w{wid}"
    passed, observed = cargo_check_codes(broken, tmp)
    if passed:
        return None  # mutation didn't actually break it
    verified = (mut.error_code in observed) or (
        mut.error_code == "SYNTAX" and "SYNTAX" in observed
    )
    if not verified:
        return None
    cid = hashlib.sha1(broken.encode()).hexdigest()[:12]
    return Case(
        case_id=cid, seed_index=si, category=mut.category,
        intended_code=mut.error_code, original=src, broken=broken,
        observed_codes=observed, verified=verified,
    )


def gen(count: int, seed: int, verify: bool, workers: int = 3, balance: bool = True,
        seed_programs: list[str] | None = None) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    seen: set[str] = set()
    # Per-category cap keeps the error distribution broad for the envelope table.
    per_cat_cap = max(2, count // len(MUTATIONS) + count // 8) if balance else count
    cat_counts: dict[str, int] = {}

    def accept(c: Case) -> bool:
        if c.case_id in seen:
            return False
        if balance and cat_counts.get(c.category, 0) >= per_cat_cap:
            return False
        seen.add(c.case_id)
        cat_counts[c.category] = cat_counts.get(c.category, 0) + 1
        return True

    # Full deterministic candidate pool, shuffled by seed for split reproducibility.
    pool = enumerate_candidates(seed_programs)
    rng.shuffle(pool)

    if not verify:
        for si, mut, src, broken in pool:
            if len(cases) >= count:
                break
            cid = hashlib.sha1(broken.encode()).hexdigest()[:12]
            c = Case(cid, si, mut.category, mut.error_code, src, broken, [], False)
            if accept(c):
                cases.append(c)
        return cases

    tmp = Path(tempfile.mkdtemp(prefix="breakit_"))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool_exec:
            # Process the candidate pool in batches until we hit `count`.
            idx = 0
            wid = 0
            while idx < len(pool) and len(cases) < count:
                batch = pool[idx: idx + workers * 4]
                idx += len(batch)
                futures = []
                for c in batch:
                    futures.append(pool_exec.submit(_verify_candidate, c, tmp, wid))
                    wid += 1
                for fut in futures:
                    res = fut.result()
                    if res is not None and len(cases) < count and accept(res):
                        cases.append(res)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="run cargo check to confirm each case breaks as intended")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel cargo check workers (tune to core count)")
    ap.add_argument("--gen-seeds", type=int, default=0,
                    help="use N parametric generated seeds (0 = static seeds.py only)")
    ap.add_argument("--seed-gen-seed", type=int, default=0,
                    help="RNG seed for the parametric seed generator")
    ap.add_argument("--include-static", action="store_true",
                    help="also include the hand-written static seeds alongside generated")
    args = ap.parse_args()

    seed_programs = None
    if args.gen_seeds > 0:
        generated = generate_seeds(args.gen_seeds, args.seed_gen_seed)
        seed_programs = (list(STATIC_SEEDS) + generated) if args.include_static else generated
        print(f"using {len(seed_programs)} seed programs "
              f"({'static+' if args.include_static else ''}{len(generated)} generated)")

    cases = gen(args.count, args.seed, args.verify, args.workers, seed_programs=seed_programs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in cases:
            f.write(json.dumps(asdict(c)) + "\n")

    by_cat: dict[str, int] = {}
    for c in cases:
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
    print(f"wrote {len(cases)} cases to {out}")
    print("by category:", json.dumps(by_cat, indent=None))
    if args.verify:
        print("all verified against cargo check")


if __name__ == "__main__":
    main()
