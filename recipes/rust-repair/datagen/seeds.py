#!/usr/bin/env python3
"""Seed corpus for rust-repair datagen.

Each seed is a small, self-contained, KNOWN-COMPILING Rust program. Mutations in
break_it.py apply labeled breakage to these. Keep them diverse across language
surfaces (types, ownership, traits, generics, collections, error handling) so the
induced error distribution is broad.

Adding seeds here is the primary way to scale corpus size without touching the
mutation engine.
"""

SEEDS: list[str] = [
    # arithmetic / primitives
    'fn main() {\n    let x: i32 = 5;\n    let y: i32 = 10;\n    println!("{}", x + y);\n}\n',
    'fn main() {\n    let a: f64 = 3.5;\n    let b: f64 = 2.0;\n    println!("{}", a * b);\n}\n',
    'fn square(n: i64) -> i64 {\n    n * n\n}\n\nfn main() {\n    println!("{}", square(9));\n}\n',
    'fn main() {\n    let flag: bool = true;\n    if flag {\n        println!("yes");\n    }\n}\n',
    # vectors / iterators
    'fn main() {\n    let v: Vec<i32> = vec![1, 2, 3];\n    let s: i32 = v.iter().sum();\n    println!("{}", s);\n}\n',
    'fn main() {\n    let v: Vec<i32> = vec![4, 5, 6];\n    let doubled: Vec<i32> = v.iter().map(|x| x * 2).collect();\n    println!("{}", doubled.len());\n}\n',
    'fn largest(v: &[i32]) -> i32 {\n    let mut m = v[0];\n    for &x in v.iter() {\n        if x > m {\n            m = x;\n        }\n    }\n    m\n}\n\nfn main() {\n    println!("{}", largest(&[3, 7, 2]));\n}\n',
    # structs / methods
    'struct Point {\n    x: i32,\n    y: i32,\n}\n\nimpl Point {\n    fn norm(&self) -> i32 {\n        self.x * self.x + self.y * self.y\n    }\n}\n\nfn main() {\n    let p = Point { x: 3, y: 4 };\n    println!("{}", p.norm());\n}\n',
    'struct Counter {\n    count: u32,\n}\n\nimpl Counter {\n    fn new() -> Counter {\n        Counter { count: 0 }\n    }\n    fn bump(&mut self) {\n        self.count += 1;\n    }\n}\n\nfn main() {\n    let mut c = Counter::new();\n    c.bump();\n    println!("{}", c.count);\n}\n',
    # strings
    'fn greet(name: &str) -> String {\n    format!("Hello, {}!", name)\n}\n\nfn main() {\n    let msg = greet("world");\n    println!("{}", msg);\n}\n',
    'fn main() {\n    let s = String::from("hello");\n    let n = s.len();\n    println!("{}", n);\n}\n',
    'fn shout(s: &str) -> String {\n    s.to_uppercase()\n}\n\nfn main() {\n    println!("{}", shout("hi"));\n}\n',
    # option / result / match
    'fn first(v: &[i32]) -> Option<i32> {\n    v.first().copied()\n}\n\nfn main() {\n    let v = vec![10, 20];\n    match first(&v) {\n        Some(x) => println!("{}", x),\n        None => println!("empty"),\n    }\n}\n',
    'fn parse(s: &str) -> Result<i32, std::num::ParseIntError> {\n    s.parse::<i32>()\n}\n\nfn main() {\n    match parse("42") {\n        Ok(n) => println!("{}", n),\n        Err(_) => println!("bad"),\n    }\n}\n',
    'fn checked_div(a: i32, b: i32) -> Option<i32> {\n    if b == 0 {\n        None\n    } else {\n        Some(a / b)\n    }\n}\n\nfn main() {\n    println!("{:?}", checked_div(10, 2));\n}\n',
    # collections (imports)
    'use std::collections::HashMap;\n\nfn main() {\n    let mut m: HashMap<String, i32> = HashMap::new();\n    m.insert("a".to_string(), 1);\n    println!("{}", m.len());\n}\n',
    'use std::collections::HashSet;\n\nfn main() {\n    let mut s: HashSet<i32> = HashSet::new();\n    s.insert(1);\n    s.insert(1);\n    println!("{}", s.len());\n}\n',
    'use std::collections::BTreeMap;\n\nfn main() {\n    let mut m: BTreeMap<i32, i32> = BTreeMap::new();\n    m.insert(1, 100);\n    println!("{}", m[&1]);\n}\n',
    # generics / traits
    'fn max_of<T: PartialOrd>(a: T, b: T) -> T {\n    if a > b {\n        a\n    } else {\n        b\n    }\n}\n\nfn main() {\n    println!("{}", max_of(3, 7));\n}\n',
    'trait Speak {\n    fn speak(&self) -> String;\n}\n\nstruct Dog;\n\nimpl Speak for Dog {\n    fn speak(&self) -> String {\n        "woof".to_string()\n    }\n}\n\nfn main() {\n    let d = Dog;\n    println!("{}", d.speak());\n}\n',
    '#[derive(Debug)]\nstruct Config {\n    name: String,\n    level: u32,\n}\n\nfn main() {\n    let c = Config {\n        name: "x".to_string(),\n        level: 2,\n    };\n    println!("{:?}", c);\n}\n',
    '#[derive(Clone, Debug)]\nstruct Item {\n    id: u32,\n}\n\nfn main() {\n    let a = Item { id: 1 };\n    let b = a.clone();\n    println!("{:?} {:?}", a, b);\n}\n',
    # ownership / borrow
    'fn main() {\n    let v = vec![1, 2, 3];\n    let r = &v;\n    println!("{}", r.len());\n}\n',
    'fn sum_all(v: &Vec<i32>) -> i32 {\n    let mut total = 0;\n    for x in v {\n        total += x;\n    }\n    total\n}\n\nfn main() {\n    let v = vec![1, 2, 3];\n    println!("{}", sum_all(&v));\n}\n',
    # closures / enums
    'fn apply<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {\n    f(x)\n}\n\nfn main() {\n    let add_one = |n| n + 1;\n    println!("{}", apply(add_one, 5));\n}\n',
    'enum Shape {\n    Circle(f64),\n    Square(f64),\n}\n\nfn area(s: &Shape) -> f64 {\n    match s {\n        Shape::Circle(r) => 3.14 * r * r,\n        Shape::Square(a) => a * a,\n    }\n}\n\nfn main() {\n    println!("{}", area(&Shape::Circle(2.0)));\n}\n',
    # more arithmetic / typed bindings
    'fn main() {\n    let count: u32 = 3;\n    let step: u32 = 2;\n    println!("{}", count * step);\n}\n',
    'fn main() {\n    let width: i64 = 8;\n    let height: i64 = 6;\n    let area: i64 = width * height;\n    println!("{}", area);\n}\n',
    'fn main() {\n    let done: bool = false;\n    let ready: bool = true;\n    println!("{} {}", done, ready);\n}\n',
    'fn triple(n: i32) -> i32 {\n    n * 3\n}\n\nfn main() {\n    let r: i32 = triple(4);\n    println!("{}", r);\n}\n',
    # more strings / &str args
    'fn banner(title: &str) -> String {\n    format!("== {} ==", title)\n}\n\nfn main() {\n    println!("{}", banner("news"));\n}\n',
    'fn repeat(word: &str) -> String {\n    word.to_string() + word\n}\n\nfn main() {\n    println!("{}", repeat("ab"));\n}\n',
    'fn initial(name: &str) -> String {\n    name.to_uppercase()\n}\n\nfn main() {\n    let x = initial("bob");\n    println!("{}", x);\n}\n',
    # more collections / imports
    'use std::collections::VecDeque;\n\nfn main() {\n    let mut q: VecDeque<i32> = VecDeque::new();\n    q.push_back(1);\n    q.push_back(2);\n    println!("{}", q.len());\n}\n',
    'use std::collections::BTreeSet;\n\nfn main() {\n    let mut s: BTreeSet<i32> = BTreeSet::new();\n    s.insert(3);\n    s.insert(1);\n    println!("{}", s.len());\n}\n',
    'use std::fmt::Write as FmtWrite;\n\nfn main() {\n    let mut out = String::new();\n    let _ = write!(out, "{}", 5);\n    println!("{}", out);\n}\n',
    # more mutable state
    'fn main() {\n    let mut total: i32 = 0;\n    for i in 1..5 {\n        total += i;\n    }\n    println!("{}", total);\n}\n',
    'fn main() {\n    let mut name = String::from("a");\n    name.push_str("bc");\n    println!("{}", name);\n}\n',
    'fn main() {\n    let mut items: Vec<i32> = Vec::new();\n    items.push(10);\n    items.push(20);\n    println!("{}", items.len());\n}\n',
    # more structs with derives / fields
    '#[derive(Debug)]\nstruct User {\n    name: String,\n    age: u32,\n}\n\nfn main() {\n    let u = User { name: "z".to_string(), age: 30 };\n    println!("{:?}", u);\n}\n',
    '#[derive(Debug, Clone)]\nstruct Vec2 {\n    x: f64,\n    y: f64,\n}\n\nfn main() {\n    let a = Vec2 { x: 1.0, y: 2.0 };\n    let b = a.clone();\n    println!("{:?} {:?}", a, b);\n}\n',
    '#[derive(Debug)]\nstruct Record {\n    id: u64,\n    active: bool,\n}\n\nfn main() {\n    let r = Record { id: 7, active: true };\n    println!("{:?}", r);\n}\n',
    'struct Rect {\n    w: i32,\n    h: i32,\n}\n\nimpl Rect {\n    fn area(&self) -> i32 {\n        self.w * self.h\n    }\n}\n\nfn main() {\n    let r = Rect { w: 4, h: 5 };\n    println!("{}", r.area());\n}\n',
    # iterators / collect with annotations
    'fn main() {\n    let nums: Vec<i32> = (1..6).collect();\n    println!("{}", nums.len());\n}\n',
    'fn main() {\n    let evens: Vec<i32> = (0..10).filter(|n| n % 2 == 0).collect();\n    println!("{}", evens.len());\n}\n',
    'fn main() {\n    let squares: Vec<i32> = vec![1, 2, 3].iter().map(|x| x * x).collect();\n    println!("{}", squares.len());\n}\n',
    # more methods / options
    'fn main() {\n    let v = vec![5, 3, 9, 1];\n    let m = v.iter().max().copied().unwrap();\n    println!("{}", m);\n}\n',
    'fn main() {\n    let s = "hello world";\n    let n = s.split(\' \').count();\n    println!("{}", n);\n}\n',
    'fn double_first(v: &[i32]) -> Option<i32> {\n    v.first().map(|x| x * 2)\n}\n\nfn main() {\n    println!("{:?}", double_first(&[4, 5]));\n}\n',
]
