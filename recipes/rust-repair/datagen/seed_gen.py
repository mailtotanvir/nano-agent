#!/usr/bin/env python3
"""Parametric seed generator for rust-repair datagen.

Programmatically synthesizes many diverse, KNOWN-COMPILING Rust programs by
filling templates with varied identifiers, integer/float/bool types, values,
method chains, struct shapes, and control flow. Each distinct fill is a distinct
compiling seed, which then multiplies through the labeled mutations in
break_it.py.

This scales the corpus to hundreds of cases without hand-writing each program.
Determinism: a seeded RNG makes the generated seed set reproducible.

Every generated program is intended to compile; break_it.py still verifies each
mutated variant with cargo, so a rare non-compiling template fill is harmless
(its mutations simply fail verification and are discarded).
"""
from __future__ import annotations

import random

INT_TYPES = ["i32", "i64", "u32", "u64", "i16", "usize"]
VAR_NAMES = ["x", "y", "n", "count", "total", "value", "num", "acc", "idx", "sum", "len", "size"]
FN_NAMES = ["compute", "calc", "process", "transform", "combine", "evaluate", "measure", "score"]
STRUCT_NAMES = ["Point", "Item", "Record", "Config", "Node", "Entry", "Cell", "Pair", "State", "Meta"]
FIELD_NAMES = ["x", "y", "id", "value", "count", "level", "size", "weight", "rank", "flag"]
STR_ARGS = ["world", "hello", "rust", "data", "input", "name", "test", "foo", "bar"]


def _tmpl_arith(r: random.Random) -> str:
    t = r.choice(INT_TYPES)
    a, b = r.sample(VAR_NAMES, 2)
    va, vb = r.randint(1, 99), r.randint(1, 99)
    op = r.choice(["+", "-", "*"])
    return (f"fn main() {{\n    let {a}: {t} = {va};\n    let {b}: {t} = {vb};\n"
            f"    println!(\"{{}}\", {a} {op} {b});\n}}\n")


def _tmpl_fn_call(r: random.Random) -> str:
    t = r.choice(INT_TYPES)
    fn = r.choice(FN_NAMES)
    p = r.choice(VAR_NAMES)
    res = r.choice(VAR_NAMES)
    k = r.randint(2, 9)
    op = r.choice(["+", "*", "-"])
    v = r.randint(1, 50)
    return (f"fn {fn}({p}: {t}) -> {t} {{\n    {p} {op} {k}\n}}\n\n"
            f"fn main() {{\n    let {res}: {t} = {fn}({v});\n    println!(\"{{}}\", {res});\n}}\n")


def _tmpl_bool(r: random.Random) -> str:
    a, b = r.sample(VAR_NAMES, 2)
    va, vb = r.choice(["true", "false"]), r.choice(["true", "false"])
    return (f"fn main() {{\n    let {a}: bool = {va};\n    let {b}: bool = {vb};\n"
            f"    println!(\"{{}} {{}}\", {a}, {b});\n}}\n")


def _tmpl_vec_iter(r: random.Random) -> str:
    v = r.choice(VAR_NAMES)
    s = r.choice([x for x in VAR_NAMES if x != v])
    nums = ", ".join(str(r.randint(1, 20)) for _ in range(r.randint(2, 4)))
    method = r.choice(["sum()", "count() as i32", "max().copied().unwrap()"])
    return (f"fn main() {{\n    let {v}: Vec<i32> = vec![{nums}];\n"
            f"    let {s}: i32 = {v}.iter().{method};\n    println!(\"{{}}\", {s});\n}}\n")


def _tmpl_vec_collect(r: random.Random) -> str:
    v = r.choice(VAR_NAMES)
    hi = r.randint(5, 15)
    op = r.choice(["* 2", "+ 1", "* {}".format(r.randint(2, 5))])
    return (f"fn main() {{\n    let {v}: Vec<i32> = (1..{hi}).map(|z| z {op}).collect();\n"
            f"    println!(\"{{}}\", {v}.len());\n}}\n")


def _tmpl_string(r: random.Random) -> str:
    fn = r.choice(FN_NAMES)
    arg = r.choice(STR_ARGS)
    method = r.choice(["to_uppercase()", "to_string()", "trim().to_string()"])
    res = r.choice(VAR_NAMES)
    return (f"fn {fn}(s: &str) -> String {{\n    s.{method}\n}}\n\n"
            f"fn main() {{\n    let {res} = {fn}(\"{arg}\");\n    println!(\"{{}}\", {res});\n}}\n")


def _tmpl_struct(r: random.Random) -> str:
    name = r.choice(STRUCT_NAMES)
    f1, f2 = r.sample(FIELD_NAMES, 2)
    t = r.choice(INT_TYPES)
    v1, v2 = r.randint(1, 50), r.randint(1, 50)
    derive = r.choice(["#[derive(Debug)]\n", "#[derive(Debug, Clone)]\n"])
    show = "{:?}" if "Debug" in derive else "{}"
    inst = r.choice(VAR_NAMES)
    if show == "{:?}":
        body = f"    println!(\"{{:?}}\", {inst});"
    else:
        body = f"    println!(\"{{}}\", {inst}.{f1});"
    return (f"{derive}struct {name} {{\n    {f1}: {t},\n    {f2}: {t},\n}}\n\n"
            f"fn main() {{\n    let {inst} = {name} {{ {f1}: {v1}, {f2}: {v2} }};\n{body}\n}}\n")


def _tmpl_struct_method(r: random.Random) -> str:
    name = r.choice(STRUCT_NAMES)
    f1, f2 = r.sample(FIELD_NAMES, 2)
    t = r.choice(["i32", "i64"])
    m = r.choice(FN_NAMES)
    op = r.choice(["+", "*"])
    inst = r.choice(VAR_NAMES)
    v1, v2 = r.randint(1, 20), r.randint(1, 20)
    return (f"struct {name} {{\n    {f1}: {t},\n    {f2}: {t},\n}}\n\n"
            f"impl {name} {{\n    fn {m}(&self) -> {t} {{\n        self.{f1} {op} self.{f2}\n    }}\n}}\n\n"
            f"fn main() {{\n    let {inst} = {name} {{ {f1}: {v1}, {f2}: {v2} }};\n"
            f"    println!(\"{{}}\", {inst}.{m}());\n}}\n")


def _tmpl_hashmap(r: random.Random) -> str:
    coll = r.choice(["HashMap", "BTreeMap"])
    m = r.choice(VAR_NAMES)
    key = r.choice(STR_ARGS)
    val = r.randint(1, 100)
    return (f"use std::collections::{coll};\n\n"
            f"fn main() {{\n    let mut {m}: {coll}<String, i32> = {coll}::new();\n"
            f"    {m}.insert(\"{key}\".to_string(), {val});\n    println!(\"{{}}\", {m}.len());\n}}\n")


def _tmpl_set(r: random.Random) -> str:
    coll = r.choice(["HashSet", "BTreeSet"])
    s = r.choice(VAR_NAMES)
    a, b = r.randint(1, 9), r.randint(1, 9)
    return (f"use std::collections::{coll};\n\n"
            f"fn main() {{\n    let mut {s}: {coll}<i32> = {coll}::new();\n"
            f"    {s}.insert({a});\n    {s}.insert({b});\n    println!(\"{{}}\", {s}.len());\n}}\n")


def _tmpl_mut_accum(r: random.Random) -> str:
    acc = r.choice(VAR_NAMES)
    t = r.choice(["i32", "i64", "u32"])
    hi = r.randint(3, 9)
    return (f"fn main() {{\n    let mut {acc}: {t} = 0;\n    for i in 1..{hi} {{\n"
            f"        {acc} += i;\n    }}\n    println!(\"{{}}\", {acc});\n}}\n")


def _tmpl_mut_string(r: random.Random) -> str:
    s = r.choice(VAR_NAMES)
    a, b = r.choice(STR_ARGS), r.choice(STR_ARGS)
    return (f"fn main() {{\n    let mut {s} = String::from(\"{a}\");\n"
            f"    {s}.push_str(\"{b}\");\n    println!(\"{{}}\", {s});\n}}\n")


def _tmpl_option_match(r: random.Random) -> str:
    fn = r.choice(FN_NAMES)
    v = r.choice(VAR_NAMES)
    nums = ", ".join(str(r.randint(1, 20)) for _ in range(2))
    return (f"fn {fn}(v: &[i32]) -> Option<i32> {{\n    v.first().copied()\n}}\n\n"
            f"fn main() {{\n    let {v} = vec![{nums}];\n    match {fn}(&{v}) {{\n"
            f"        Some(z) => println!(\"{{}}\", z),\n        None => println!(\"empty\"),\n    }}\n}}\n")


def _tmpl_generic(r: random.Random) -> str:
    fn = r.choice(FN_NAMES)
    a, b = r.randint(1, 50), r.randint(1, 50)
    return (f"fn {fn}<T: PartialOrd>(a: T, b: T) -> T {{\n    if a > b {{\n        a\n    }} else {{\n        b\n    }}\n}}\n\n"
            f"fn main() {{\n    println!(\"{{}}\", {fn}({a}, {b}));\n}}\n")


TEMPLATES = [
    _tmpl_arith, _tmpl_fn_call, _tmpl_bool, _tmpl_vec_iter, _tmpl_vec_collect,
    _tmpl_string, _tmpl_struct, _tmpl_struct_method, _tmpl_hashmap, _tmpl_set,
    _tmpl_mut_accum, _tmpl_mut_string, _tmpl_option_match, _tmpl_generic,
]


def generate_seeds(n: int, seed: int = 0) -> list[str]:
    """Generate up to `n` unique compiling seed programs deterministically."""
    r = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        tmpl = r.choice(TEMPLATES)
        prog = tmpl(r)
        if prog in seen:
            continue
        seen.add(prog)
        out.append(prog)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    seeds = generate_seeds(args.n, args.seed)
    print(f"generated {len(seeds)} unique seeds")
    if args.print:
        for i, s in enumerate(seeds[:5]):
            print(f"--- seed {i} ---\n{s}")
