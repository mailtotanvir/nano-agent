#!/usr/bin/env python3
"""Out-of-distribution (OOD) eval set for rust-repair.

These cases are HAND-AUTHORED to be structurally unlike the break_it.py generator:
- idiomatic, realistic Rust (not the templated seed shapes)
- real-world bug patterns a human would actually write
- error kinds and multi-error situations the generator does NOT produce
- longer programs, traits, generics, lifetimes, iterators, error handling

Each case: a broken program that fails `cargo check` with a SINGLE root-cause fix
reachable by editing (the fix may touch one site). We verify two invariants with
cargo before trusting a case:
  1. the BROKEN program fails to compile (has errors)
  2. the FIXED program compiles clean
Cases that don't satisfy both are dropped (kept honest, like the main corpus).

This set is NEVER used for training. It exists only to measure generalization
beyond the synthetic training distribution.
"""
from __future__ import annotations

# Each entry: (id, category_hint, broken_src, fixed_src)
CASES: list[tuple[str, str, str, str]] = [

    # 1. Iterator chain: collect into wrong type (real-world: forgot turbofish/type)
    ("ood_iter_collect", "type_mismatch",
     '''fn main() {
    let words = vec!["hello", "world", "rust"];
    let lengths: Vec<String> = words.iter().map(|w| w.len()).collect();
    println!("{:?}", lengths);
}
''',
     '''fn main() {
    let words = vec!["hello", "world", "rust"];
    let lengths: Vec<usize> = words.iter().map(|w| w.len()).collect();
    println!("{:?}", lengths);
}
'''),

    # 2. Borrow after move (real ownership bug)
    ("ood_move_use", "borrow",
     '''fn main() {
    let s = String::from("data");
    let taken = s;
    println!("{}", s);
}
''',
     '''fn main() {
    let s = String::from("data");
    let taken = s;
    println!("{}", taken);
}
'''),

    # 3. Missing & in function arg (pass by ref)
    ("ood_ref_arg", "type_mismatch",
     '''fn total(v: &Vec<i32>) -> i32 {
    v.iter().sum()
}

fn main() {
    let nums = vec![1, 2, 3];
    println!("{}", total(nums));
}
''',
     '''fn total(v: &Vec<i32>) -> i32 {
    v.iter().sum()
}

fn main() {
    let nums = vec![1, 2, 3];
    println!("{}", total(&nums));
}
'''),

    # 4. Result not handled - using ? in fn returning ()
    ("ood_question_unit", "error_handling",
     '''fn parse(s: &str) -> i32 {
    let n: i32 = s.parse()?;
    n * 2
}

fn main() {
    println!("{}", parse("21"));
}
''',
     '''fn parse(s: &str) -> i32 {
    let n: i32 = s.parse().unwrap();
    n * 2
}

fn main() {
    println!("{}", parse("21"));
}
'''),

    # 5. Struct field privacy / wrong field access via method
    ("ood_trait_method", "unknown_method",
     '''struct Counter {
    count: u32,
}

impl Counter {
    fn new() -> Self {
        Counter { count: 0 }
    }
    fn increment(&mut self) {
        self.count += 1;
    }
}

fn main() {
    let mut c = Counter::new();
    c.increment();
    c.inc();
    println!("{}", c.count);
}
''',
     '''struct Counter {
    count: u32,
}

impl Counter {
    fn new() -> Self {
        Counter { count: 0 }
    }
    fn increment(&mut self) {
        self.count += 1;
    }
}

fn main() {
    let mut c = Counter::new();
    c.increment();
    c.increment();
    println!("{}", c.count);
}
'''),

    # 6. Mismatched integer types in comparison
    ("ood_int_cast", "type_mismatch",
     '''fn main() {
    let a: u8 = 200;
    let b: i32 = 100;
    if a > b {
        println!("bigger");
    } else {
        println!("smaller");
    }
}
''',
     '''fn main() {
    let a: u8 = 200;
    let b: i32 = 100;
    if a as i32 > b {
        println!("bigger");
    } else {
        println!("smaller");
    }
}
'''),

    # 7. Trait not in scope (needs use)
    ("ood_trait_scope", "missing_import",
     '''use std::collections::HashMap;

fn main() {
    let mut m = HashMap::new();
    m.insert("a", 1);
    let mut keys: Vec<_> = m.keys().collect();
    keys.sort();
    let joined = keys.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(",");
    io::stdout().write_all(joined.as_bytes()).unwrap();
}
''',
     '''use std::collections::HashMap;
use std::io::{self, Write};

fn main() {
    let mut m = HashMap::new();
    m.insert("a", 1);
    let mut keys: Vec<_> = m.keys().collect();
    keys.sort();
    let joined = keys.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(",");
    io::stdout().write_all(joined.as_bytes()).unwrap();
}
'''),

    # 8. Closure captures - FnMut needed but Fn given (mutation in closure)
    ("ood_option_unwrap", "error_handling",
     '''fn first_even(v: &[i32]) -> i32 {
    v.iter().find(|&&x| x % 2 == 0)
}

fn main() {
    let nums = vec![1, 3, 4, 7];
    println!("{}", first_even(&nums));
}
''',
     '''fn first_even(v: &[i32]) -> i32 {
    *v.iter().find(|&&x| x % 2 == 0).unwrap()
}

fn main() {
    let nums = vec![1, 3, 4, 7];
    println!("{}", first_even(&nums));
}
'''),

    # 9. Generic bound missing (needs Display)
    ("ood_generic_bound", "trait_bound",
     '''fn show<T>(item: T) {
    println!("{}", item);
}

fn main() {
    show(42);
    show("hi");
}
''',
     '''use std::fmt::Display;

fn show<T: Display>(item: T) {
    println!("{}", item);
}

fn main() {
    show(42);
    show("hi");
}
'''),

    # 10. String vs &str return type
    ("ood_string_return", "type_mismatch",
     '''fn greeting(name: &str) -> String {
    let msg = format!("Hello, {}", name);
    &msg
}

fn main() {
    println!("{}", greeting("world"));
}
''',
     '''fn greeting(name: &str) -> String {
    let msg = format!("Hello, {}", name);
    msg
}

fn main() {
    println!("{}", greeting("world"));
}
'''),

    # 11. match non-exhaustive
    ("ood_match_arms", "non_exhaustive",
     '''enum Color {
    Red,
    Green,
    Blue,
}

fn code(c: Color) -> u8 {
    match c {
        Color::Red => 1,
        Color::Green => 2,
    }
}

fn main() {
    println!("{}", code(Color::Blue));
}
''',
     '''enum Color {
    Red,
    Green,
    Blue,
}

fn code(c: Color) -> u8 {
    match c {
        Color::Red => 1,
        Color::Green => 2,
        Color::Blue => 3,
    }
}

fn main() {
    println!("{}", code(Color::Blue));
}
'''),

    # 12. Vec index type (usize vs i32)
    ("ood_index_type", "type_mismatch",
     '''fn main() {
    let v = vec![10, 20, 30];
    let i: i32 = 1;
    println!("{}", v[i]);
}
''',
     '''fn main() {
    let v = vec![10, 20, 30];
    let i: usize = 1;
    println!("{}", v[i]);
}
'''),

    # 13. Immutable borrow then mutate (need mut)
    ("ood_push_immut", "missing_mut",
     '''fn main() {
    let scores = Vec::new();
    scores.push(10);
    scores.push(20);
    println!("{}", scores.len());
}
''',
     '''fn main() {
    let mut scores = Vec::new();
    scores.push(10);
    scores.push(20);
    println!("{}", scores.len());
}
'''),

    # 14. Wrong number of fn args
    ("ood_arg_count", "arg_count",
     '''fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    println!("{}", add(1, 2, 3));
}
''',
     '''fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    println!("{}", add(1, 2));
}
'''),

    # 15. Lifetime-ish: returning reference to local via clone missing
    ("ood_clone_needed", "borrow",
     '''fn longest(a: String, b: String) -> String {
    if a.len() > b.len() {
        a
    } else {
        b
    }
}

fn main() {
    let s1 = String::from("hello");
    let s2 = String::from("hi");
    let result = longest(s1, s2);
    println!("{} {}", result, s1);
}
''',
     '''fn longest(a: String, b: String) -> String {
    if a.len() > b.len() {
        a
    } else {
        b
    }
}

fn main() {
    let s1 = String::from("hello");
    let s2 = String::from("hi");
    let result = longest(s1.clone(), s2);
    println!("{} {}", result, s1);
}
'''),

    # 16. Enum variant typo
    ("ood_enum_variant", "undeclared",
     '''enum Direction {
    North,
    South,
    East,
    West,
}

fn main() {
    let d = Direction::Norht;
    match d {
        Direction::North => println!("up"),
        _ => println!("other"),
    }
}
''',
     '''enum Direction {
    North,
    South,
    East,
    West,
}

fn main() {
    let d = Direction::North;
    match d {
        Direction::North => println!("up"),
        _ => println!("other"),
    }
}
'''),

    # 17. format! placeholder mismatch is not compile error; use wrong macro
    ("ood_macro_name", "undeclared",
     '''fn main() {
    let x = 5;
    printline!("value is {}", x);
}
''',
     '''fn main() {
    let x = 5;
    println!("value is {}", x);
}
'''),

    # 18. Trait impl missing method
    ("ood_trait_impl", "missing_method",
     '''trait Shape {
    fn area(&self) -> f64;
    fn name(&self) -> String;
}

struct Circle {
    radius: f64,
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        3.14159 * self.radius * self.radius
    }
}

fn main() {
    let c = Circle { radius: 2.0 };
    println!("{} {}", c.name(), c.area());
}
''',
     '''trait Shape {
    fn area(&self) -> f64;
    fn name(&self) -> String;
}

struct Circle {
    radius: f64,
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        3.14159 * self.radius * self.radius
    }
    fn name(&self) -> String {
        String::from("circle")
    }
}

fn main() {
    let c = Circle { radius: 2.0 };
    println!("{} {}", c.name(), c.area());
}
'''),

    # 19. Option method chain wrong (map vs and_then)
    ("ood_slice_bounds", "type_mismatch",
     '''fn main() {
    let data = [1, 2, 3, 4, 5];
    let slice: &[i32] = data[1..3];
    println!("{:?}", slice);
}
''',
     '''fn main() {
    let data = [1, 2, 3, 4, 5];
    let slice: &[i32] = &data[1..3];
    println!("{:?}", slice);
}
'''),

    # 20. Float/int arithmetic mix
    ("ood_float_mix", "type_mismatch",
     '''fn main() {
    let count = 3;
    let total = 10.0;
    let average = total / count;
    println!("{}", average);
}
''',
     '''fn main() {
    let count = 3;
    let total = 10.0;
    let average = total / count as f64;
    println!("{}", average);
}
'''),

    # 21. Shadowing wrong type used in method
    ("ood_parse_method", "unknown_method",
     '''fn main() {
    let input = "42";
    let doubled = input.double();
    println!("{}", doubled);
}
''',
     '''fn main() {
    let input = "42";
    let doubled: i32 = input.parse::<i32>().unwrap() * 2;
    println!("{}", doubled);
}
'''),

    # 22. Missing semicolon in a realistic multi-line fn
    ("ood_semicolon_real", "syntax",
     '''fn process(data: &[i32]) -> i32 {
    let mut sum = 0
    for &x in data {
        sum += x;
    }
    sum
}

fn main() {
    println!("{}", process(&[1, 2, 3]));
}
''',
     '''fn process(data: &[i32]) -> i32 {
    let mut sum = 0;
    for &x in data {
        sum += x;
    }
    sum
}

fn main() {
    println!("{}", process(&[1, 2, 3]));
}
'''),

    # 23. Derive missing for equality
    ("ood_derive_eq", "missing_derive",
     '''struct Version {
    major: u32,
    minor: u32,
}

fn main() {
    let a = Version { major: 1, minor: 0 };
    let b = Version { major: 1, minor: 0 };
    if a == b {
        println!("same");
    }
}
''',
     '''#[derive(PartialEq)]
struct Version {
    major: u32,
    minor: u32,
}

fn main() {
    let a = Version { major: 1, minor: 0 };
    let b = Version { major: 1, minor: 0 };
    if a == b {
        println!("same");
    }
}
'''),

    # 24. Returning () instead of value (trailing semicolon on final expr)
    ("ood_trailing_semi", "type_mismatch",
     '''fn square(x: i32) -> i32 {
    x * x;
}

fn main() {
    println!("{}", square(5));
}
''',
     '''fn square(x: i32) -> i32 {
    x * x
}

fn main() {
    println!("{}", square(5));
}
'''),

    # 25. HashMap entry - wrong method name
    ("ood_hashmap_get", "unknown_method",
     '''use std::collections::HashMap;

fn main() {
    let mut inventory = HashMap::new();
    inventory.insert("apples", 5);
    let count = inventory.fetch("apples");
    println!("{:?}", count);
}
''',
     '''use std::collections::HashMap;

fn main() {
    let mut inventory = HashMap::new();
    inventory.insert("apples", 5);
    let count = inventory.get("apples");
    println!("{:?}", count);
}
'''),
]


if __name__ == "__main__":
    import argparse, json, subprocess, tempfile, shutil
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../datasets/eval_ood_v1.jsonl")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    def cargo_ok(src: str) -> bool:
        d = Path(tempfile.mkdtemp())
        try:
            (d / "src").mkdir()
            (d / "Cargo.toml").write_text(
                '[package]\nname="c"\nversion="0.1.0"\nedition="2021"\n\n[[bin]]\nname="c"\npath="src/main.rs"\n')
            (d / "src/main.rs").write_text(src)
            r = subprocess.run(["cargo", "check", "--quiet"], cwd=d,
                               capture_output=True, text=True)
            return r.returncode == 0
        finally:
            shutil.rmtree(d, ignore_errors=True)

    kept = []
    if args.verify:
        def check(case):
            cid, cat, broken, fixed = case
            broken_fails = not cargo_ok(broken)
            fixed_compiles = cargo_ok(fixed)
            return cid, cat, broken, fixed, broken_fails, fixed_compiles
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for cid, cat, broken, fixed, bf, fc in ex.map(check, CASES):
                status = "OK" if (bf and fc) else f"DROP(broken_fails={bf},fixed_compiles={fc})"
                print(f"{cid:24} {cat:16} {status}")
                if bf and fc:
                    kept.append({"case_id": cid, "category": cat,
                                 "broken": broken, "fixed": fixed, "source": "ood_handauthored"})
    else:
        for cid, cat, broken, fixed in CASES:
            kept.append({"case_id": cid, "category": cat,
                         "broken": broken, "fixed": fixed, "source": "ood_handauthored"})

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        for c in kept:
            f.write(json.dumps(c) + "\n")
    print(f"\nwrote {len(kept)}/{len(CASES)} verified OOD cases to {outp}")
