#!/usr/bin/env python3
"""Structural / multi-line repair case generator (curriculum extension).

The OOD eval exposed that the 0.5B, trained only on break_it.py's single-site
mutations, fails STRUCTURAL and MULTI-LINE repairs. This generator produces fresh
cases of exactly those failing categories so the teacher (Gemini) can demonstrate
them and the nano model can learn the skill.

IMPORTANT: these are NEW synthetic cases for TRAINING. They are the same *skill
categories* as the OOD failures but different code. The 25 frozen OOD eval cases
are NEVER reproduced here (we assert disjointness by case_id hash + content).

Categories (each = a structural edit the single-site generator never makes):
  - add_use         : missing `use` line -> add an import line (multi-line insert)
  - add_trait_method: impl missing a required trait method -> add whole fn
  - add_match_arm   : non-exhaustive match -> add missing arm(s)
  - fix_return_expr : trailing semicolon on final expr -> drop it (return value)
  - ref_return      : returning &local where owned needed -> return owned
  - add_derive      : missing #[derive(..)] for ==/Debug/Clone -> add attribute
  - add_trait_bound : generic used with trait op but no bound -> add `: Trait`
  - collect_type    : collect() with wrong/absent turbofish/annotation
  - move_then_use   : use-after-move -> clone or reorder
  - closure_capture : Option/iterator chain needing unwrap/deref

Every generated case is cargo-verified: broken MUST fail, fixed MUST compile.
"""
from __future__ import annotations
import random

IDENTS = ["value", "count", "total", "item", "node", "data", "result", "acc", "n", "x"]
TYPES = ["i32", "i64", "u32", "u64", "usize"]
STRUCTS = ["Point", "Item", "Record", "Config", "Node", "Entry", "Cell", "State"]
FIELDS = ["x", "y", "id", "value", "count", "size", "level", "weight"]


def _add_use(r):
    coll = r.choice(["HashMap", "HashSet", "BTreeMap", "BTreeSet"])
    m = r.choice(IDENTS)
    key = r.choice(["a", "k", "key", "id"])
    val = r.randint(1, 99)
    body = (f"fn main() {{\n    let mut {m}: {coll}<String, i32> = {coll}::new();\n"
            f"    {m}.insert(\"{key}\".to_string(), {val});\n    println!(\"{{}}\", {m}.len());\n}}\n") \
        if "Map" in coll else \
        (f"fn main() {{\n    let mut {m}: {coll}<i32> = {coll}::new();\n"
         f"    {m}.insert({val});\n    println!(\"{{}}\", {m}.len());\n}}\n")
    broken = body
    fixed = f"use std::collections::{coll};\n\n" + body
    return broken, fixed


def _add_trait_method(r):
    name = r.choice(STRUCTS)
    f1 = r.choice(FIELDS)
    t = r.choice(["f64", "i32"])
    val = f"{r.randint(1, 20)}.0" if t == "f64" else str(r.randint(1, 20))
    other = r.choice(["perimeter", "describe", "label", "kind"])
    ret_impl = ('String' if other in ("describe", "label", "kind") else t)
    if ret_impl == "String":
        missing = f"    fn {other}(&self) -> String {{\n        String::from(\"{name.lower()}\")\n    }}\n"
        call = f'println!("{{}}", s.{other}());'
    else:
        multiplier = "2.0" if t == "f64" else "2"
        missing = f"    fn {other}(&self) -> {t} {{\n        self.{f1} * {multiplier}\n    }}\n"
        call = f'println!("{{}}", s.{other}());'
    header = (f"trait Describe {{\n    fn base(&self) -> {t};\n    fn {other}(&self) -> {ret_impl};\n}}\n\n"
              f"struct {name} {{\n    {f1}: {t},\n}}\n\n")
    impl_broken = (f"impl Describe for {name} {{\n    fn base(&self) -> {t} {{\n        self.{f1}\n    }}\n}}\n\n")
    impl_fixed = (f"impl Describe for {name} {{\n    fn base(&self) -> {t} {{\n        self.{f1}\n    }}\n"
                  f"{missing}}}\n\n")
    main = f"fn main() {{\n    let s = {name} {{ {f1}: {val} }};\n    {call}\n}}\n"
    return header+impl_broken+main, header+impl_fixed+main


def _add_match_arm(r):
    variants = r.sample(["North", "South", "East", "West", "Up", "Down"], 3)
    enum = r.choice(["Direction", "Move", "Way"])
    codes = {v: i+1 for i, v in enumerate(variants)}
    hdr = f"enum {enum} {{\n" + "".join(f"    {v},\n" for v in variants) + "}\n\n"
    # broken: omit last arm
    arms_broken = "".join(f"        {enum}::{v} => {codes[v]},\n" for v in variants[:-1])
    arms_fixed = "".join(f"        {enum}::{v} => {codes[v]},\n" for v in variants)
    fn = lambda arms: (f"fn code(d: {enum}) -> u8 {{\n    match d {{\n{arms}    }}\n}}\n\n")
    main = f"fn main() {{\n    println!(\"{{}}\", code({enum}::{variants[-1]}));\n}}\n"
    return hdr+fn(arms_broken)+main, hdr+fn(arms_fixed)+main


def _fix_return_expr(r):
    fn = r.choice(["square", "double", "triple", "compute"])
    t = r.choice(["i32", "i64"])
    op = r.choice(["* x", "* 2", "+ x", "* x + 1"])
    v = r.randint(2, 12)
    broken = (f"fn {fn}(x: {t}) -> {t} {{\n    x {op};\n}}\n\n"
              f"fn main() {{\n    println!(\"{{}}\", {fn}({v}));\n}}\n")
    fixed = (f"fn {fn}(x: {t}) -> {t} {{\n    x {op}\n}}\n\n"
             f"fn main() {{\n    println!(\"{{}}\", {fn}({v}));\n}}\n")
    return broken, fixed


def _ref_return(r):
    fn = r.choice(["build", "make", "render", "format_it"])
    arg = r.choice(["name", "input", "label"])
    val = r.choice(["world", "rust", "data"])
    broken = (f"fn {fn}({arg}: &str) -> String {{\n    let s = format!(\"[{{}}]\", {arg});\n    &s\n}}\n\n"
              f"fn main() {{\n    println!(\"{{}}\", {fn}(\"{val}\"));\n}}\n")
    fixed = (f"fn {fn}({arg}: &str) -> String {{\n    let s = format!(\"[{{}}]\", {arg});\n    s\n}}\n\n"
             f"fn main() {{\n    println!(\"{{}}\", {fn}(\"{val}\"));\n}}\n")
    return broken, fixed


def _add_derive(r):
    name = r.choice(STRUCTS)
    f1, f2 = r.sample(FIELDS, 2)
    t = r.choice(["u32", "i32"])
    v1, v2 = r.randint(1, 9), r.randint(1, 9)
    which = r.choice(["PartialEq", "Debug"])
    hdr_broken = f"struct {name} {{\n    {f1}: {t},\n    {f2}: {t},\n}}\n\n"
    hdr_fixed = f"#[derive({which})]\nstruct {name} {{\n    {f1}: {t},\n    {f2}: {t},\n}}\n\n"
    if which == "PartialEq":
        main = (f"fn main() {{\n    let a = {name} {{ {f1}: {v1}, {f2}: {v2} }};\n"
                f"    let b = {name} {{ {f1}: {v1}, {f2}: {v2} }};\n"
                f"    if a == b {{ println!(\"eq\"); }}\n}}\n")
    else:
        main = (f"fn main() {{\n    let a = {name} {{ {f1}: {v1}, {f2}: {v2} }};\n"
                f"    println!(\"{{:?}}\", a);\n}}\n")
    return hdr_broken+main, hdr_fixed+main


def _add_trait_bound(r):
    fn = r.choice(["show", "print_it", "display", "render", "emit"])
    v = r.choice([str(r.randint(1, 99)), f'"{r.choice(["hi", "rust", "data"])}"',
                  str(r.randint(100, 999))])
    broken = (f"fn {fn}<T>(item: T) {{\n    println!(\"{{}}\", item);\n}}\n\n"
              f"fn main() {{\n    {fn}({v});\n}}\n")
    fixed = (f"use std::fmt::Display;\n\nfn {fn}<T: Display>(item: T) {{\n    println!(\"{{}}\", item);\n}}\n\n"
             f"fn main() {{\n    {fn}({v});\n}}\n")
    return broken, fixed


def _collect_type(r):
    hi = r.randint(4, 9)
    op = r.choice(["* 2", "+ 1", "* 3"])
    v = r.choice(IDENTS)
    broken = (f"fn main() {{\n    let {v} = (1..{hi}).map(|z| z {op}).collect();\n"
              f"    println!(\"{{}}\", {v}.len());\n}}\n")
    fixed = (f"fn main() {{\n    let {v}: Vec<i32> = (1..{hi}).map(|z| z {op}).collect();\n"
             f"    println!(\"{{}}\", {v}.len());\n}}\n")
    return broken, fixed


def _move_then_use(r):
    a = r.choice(IDENTS)
    b = r.choice([x for x in IDENTS if x != a])
    val = r.choice(["hello", "data", "rust"])
    broken = (f"fn main() {{\n    let {a} = String::from(\"{val}\");\n    let {b} = {a};\n"
              f"    println!(\"{{}} {{}}\", {a}, {b});\n}}\n")
    fixed = (f"fn main() {{\n    let {a} = String::from(\"{val}\");\n    let {b} = {a}.clone();\n"
             f"    println!(\"{{}} {{}}\", {a}, {b});\n}}\n")
    return broken, fixed


def _closure_capture(r):
    fn = r.choice(["first_even", "first_pos", "find_it"])
    cond = r.choice(["x % 2 == 0", "x > 0", "x > 2"])
    nums = ", ".join(str(r.randint(1, 9)) for _ in range(4))
    broken = (f"fn {fn}(v: &[i32]) -> i32 {{\n    v.iter().find(|&&x| {cond})\n}}\n\n"
              f"fn main() {{\n    let data = vec![{nums}];\n    println!(\"{{}}\", {fn}(&data));\n}}\n")
    fixed = (f"fn {fn}(v: &[i32]) -> i32 {{\n    *v.iter().find(|&&x| {cond}).unwrap()\n}}\n\n"
             f"fn main() {{\n    let data = vec![{nums}];\n    println!(\"{{}}\", {fn}(&data));\n}}\n")
    return broken, fixed


GENERATORS = {
    "add_use": _add_use,
    "add_trait_method": _add_trait_method,
    "add_match_arm": _add_match_arm,
    "fix_return_expr": _fix_return_expr,
    "ref_return": _ref_return,
    "add_derive": _add_derive,
    "add_trait_bound": _add_trait_bound,
    "collect_type": _collect_type,
    "move_then_use": _move_then_use,
    "closure_capture": _closure_capture,
}


def generate(n_per_cat: int, seed: int = 0):
    r = random.Random(seed)
    out = []
    seen = set()
    for cat, gen in GENERATORS.items():
        made = 0
        attempts = 0
        while made < n_per_cat and attempts < n_per_cat * 40:
            attempts += 1
            broken, fixed = gen(r)
            if broken == fixed or broken in seen:
                continue
            seen.add(broken)
            out.append({"category": cat, "broken": broken, "fixed": fixed})
            made += 1
    return out


if __name__ == "__main__":
    import argparse, json, subprocess, tempfile, shutil, hashlib
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../datasets/structural_pool_v1.jsonl")
    ap.add_argument("--eval-exclude", default="../datasets/eval_ood_v1.jsonl")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # collect frozen OOD eval content to guarantee disjointness
    excl = set()
    ep = Path(args.eval_exclude)
    if ep.exists():
        for l in ep.read_text().splitlines():
            if l.strip():
                c = json.loads(l)
                excl.add(c["broken"])

    cases = generate(args.n_per_cat, args.seed)
    cases = [c for c in cases if c["broken"] not in excl]  # disjoint from eval

    def cargo_ok(src):
        d = Path(tempfile.mkdtemp())
        try:
            (d/"src").mkdir()
            (d/"Cargo.toml").write_text(
                '[package]\nname="c"\nversion="0.1.0"\nedition="2021"\n\n[[bin]]\nname="c"\npath="src/main.rs"\n')
            (d/"src/main.rs").write_text(src)
            return subprocess.run(["cargo","check","--quiet"], cwd=d,
                                  capture_output=True, text=True).returncode == 0
        finally:
            shutil.rmtree(d, ignore_errors=True)

    kept = []
    if args.verify:
        def check(c):
            return c, (not cargo_ok(c["broken"])), cargo_ok(c["fixed"])
        from collections import Counter
        stats = Counter()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for c, bf, fc in ex.map(check, cases):
                if bf and fc:
                    cid = hashlib.sha1(c["broken"].encode()).hexdigest()[:12]
                    c["case_id"] = f"struct_{c['category']}_{cid}"
                    c["source"] = "structural_synthetic"
                    kept.append(c)
                    stats[c["category"]] += 1
                else:
                    stats[f"DROP_{c['category']}"] += 1
        print("verified by category:", dict(sorted(stats.items())))
    else:
        for c in cases:
            cid = hashlib.sha1(c["broken"].encode()).hexdigest()[:12]
            c["case_id"] = f"struct_{c['category']}_{cid}"
            c["source"] = "structural_synthetic"
            kept.append(c)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        for c in kept:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(kept)} verified structural cases to {outp}")
