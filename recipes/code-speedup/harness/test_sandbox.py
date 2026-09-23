"""Real subprocess tests: no simulated isolation or candidate outputs."""
import importlib
import json
import os
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("body", [
    "    out = 0\n    for i in range(600):\n        out = [out]\n    return out",
    "    return (1, 2)",
    "    return {1: 'one'}",
])
def test_outputs_must_be_shallow_strict_json(body):
    sandbox = importlib.import_module("harness.sandbox")
    result = sandbox.run_candidate("def f():\n" + body, "f", [[]])
    assert not result.ok
    assert result.status == "output_error", result


@pytest.mark.parametrize("tool_memory_mb,expected", [(32, False), (1024, True)])
def test_cachegrind_has_separate_kernel_memory_cap(tool_memory_mb, expected):
    sandbox = importlib.import_module("harness.sandbox")
    binary = Path(os.environ.get("VALGRIND", "/usr/bin/valgrind"))
    if not binary.is_file():
        pytest.skip("real cachegrind unavailable")
    lib_dir = os.environ.get("VALGRIND_LIB")
    config = sandbox.CachegrindConfig(binary, Path(lib_dir) if lib_dir else None,
                                      tool_memory_mb=tool_memory_mb)
    result = sandbox.run_candidate("def f():\n    return 1\n", "f", [[]],
                                   cachegrind=config, timeout_s=20, cpu_seconds=10)
    assert result.ok is expected, result
    if not expected:
        assert result.profile is None
        assert result.outputs == []


@pytest.mark.parametrize("expression", ["float('nan')", "float('inf')", "set([1])"])
def test_non_json_outputs_are_structured_failures(expression):
    sandbox = importlib.import_module("harness.sandbox")
    result = sandbox.run_candidate(f"def f():\n    return {expression}\n", "f", [[]])
    assert not result.ok
    assert result.status == "output_error", result


def test_real_namespace_filesystem_and_environment(monkeypatch):
    sandbox = importlib.import_module("harness.sandbox")
    monkeypatch.setenv("SANDBOX_SENTINEL", "must-not-leak")
    probe = """import json, os, pathlib, socket
p = pathlib.Path
print(json.dumps({
    'home': p('/home').exists(), 'etc': p('/etc').exists(),
    'harness': p('/private/workspace/recipes/code-speedup/harness/sandbox.py').exists(),
    'env': os.environ.get('SANDBOX_SENTINEL'), 'pid': os.getpid(),
    'net': os.readlink('/proc/self/ns/net'),
    'interfaces': socket.if_nameindex(),
    'tmp': p('/tmp').is_dir(),
}))
"""
    command = sandbox.isolation_command() + ["--", "/usr/bin/python3", "-s", "-S", "-c", probe]
    status, output, code = sandbox._capture(command, "", 2, 4096)
    assert status == "ok" and code == 0, (status, output, code)
    state = json.loads(output["stdout"])
    assert state == {"home": False, "etc": False, "harness": False, "env": None, "pid": 1,
                     "net": state["net"], "interfaces": [[1, "lo"]], "tmp": True}
    assert state["net"] != os.readlink('/proc/self/ns/net')


@pytest.mark.parametrize("changes", [
    {"timeout_s": 0}, {"timeout_s": float("nan")}, {"memory_mb": -1},
    {"cpu_seconds": 0}, {"max_output_bytes": -1}, {"inputs": [1]},
    {"inputs": [[float("inf")]]}, {"source": "x" * 65537},
    {"inputs": [["x" * 1100000]]}, {"function_name": "not a name"},
])
def test_invalid_requests_are_rejected(changes):
    sandbox = importlib.import_module("harness.sandbox")
    request = {"source": "def f():\n    return 1\n", "function_name": "f", "inputs": [[]]}
    request.update(changes)
    result = sandbox.run_candidate(**request)
    assert not result.ok
    assert result.status == "invalid_request", result


def test_cpu_limit_is_independent_of_wall_limit():
    sandbox = importlib.import_module("harness.sandbox")
    result = sandbox.run_candidate("def spin():\n    while True:\n        pass\n", "spin", [[]],
                                   timeout_s=5, cpu_seconds=1)
    assert not result.ok
    assert result.status == "resource_limit", result


@pytest.mark.parametrize("binary", ["/nonexistent/bwrap", "/usr/bin/false"])
def test_isolation_failure_never_falls_back(monkeypatch, binary):
    sandbox = importlib.import_module("harness.sandbox")
    monkeypatch.setattr(sandbox, "BWRAP", binary, raising=False)
    result = sandbox.run_candidate("def f():\n    return 7\n", "f", [[]])
    assert not result.ok
    assert result.status == "sandbox_error"
    assert result.outputs == []


def test_cachegrind_uses_same_sandbox_and_returns_profile():
    sandbox = importlib.import_module("harness.sandbox")
    binary = Path(os.environ.get("VALGRIND", "/usr/bin/valgrind"))
    if not binary.is_file():
        pytest.skip("real cachegrind not installed; set VALGRIND and VALGRIND_LIB")
    lib_dir = os.environ.get("VALGRIND_LIB")
    config = sandbox.CachegrindConfig(binary, Path(lib_dir) if lib_dir else None)
    result = sandbox.run_candidate("def f(xs):\n    return sorted(xs)\n", "f", [[[3, 1, 2]]],
                                   cachegrind=config, timeout_s=20, cpu_seconds=10)
    assert result.ok, result
    assert result.outputs == [[1, 2, 3]]
    assert "events: Ir" in result.profile
    assert "summary:" in result.profile


def test_output_flood_is_bounded():
    sandbox = importlib.import_module("harness.sandbox")
    result = sandbox.run_candidate("def flood():\n    return 'x' * 2000000\n", "flood", [[]],
                                   max_output_bytes=4096)
    assert not result.ok
    assert result.status == "output_limit", result
    assert result.outputs == []


def test_algorithmic_builtins_and_methods():
    sandbox = importlib.import_module("harness.sandbox")
    result = sandbox.run_candidate(
        "def f(xs):\n    out = []\n    for i in range(len(xs)):\n        out.append(abs(xs[i]))\n    return sorted(out)\n",
        "f", [[[-3, 2, -1]]],
    )
    assert result.ok, result
    assert result.outputs == [[1, 2, 3]]


@pytest.mark.parametrize("source", [
    "def f():\n    import socket\n    return socket.socket()",
    "def f():\n    return open('/etc/passwd').read()",
    "def f():\n    import os\n    return os.fork()",
    "def f():\n    return ().__class__.__base__.__subclasses__()",
    "def f():\n    return f.__globals__",
    "def f():\n    return __import__('subprocess').run(['true'])",
    "def f(x=open('/work/cachegrind.out')):\n    return 1",
    "x = 1\ndef f():\n    return x",
])
def test_unsafe_syntax_is_denied(source):
    sandbox = importlib.import_module("harness.sandbox")
    result = sandbox.run_candidate(source, "f", [[]])
    assert not result.ok
    assert result.status == "policy_error", result


def test_memory_abuse_is_bounded():
    sandbox = importlib.import_module("harness.sandbox")
    result = sandbox.run_candidate("def allocate():\n    return [0] * 100000000\n", "allocate", [[]],
                                   memory_mb=64)
    assert not result.ok
    assert result.status == "memory_limit"


def test_infinite_loop_hits_wall_timeout():
    sandbox = importlib.import_module("harness.sandbox")
    started = time.monotonic()
    result = sandbox.run_candidate("def spin():\n    while True:\n        pass\n", "spin", [[]],
                                   timeout_s=0.15, cpu_seconds=2)
    assert not result.ok
    assert result.status == "timeout"
    assert time.monotonic() - started < 3


def test_valid_pure_function():
    sandbox = importlib.import_module("harness.sandbox")
    result = sandbox.run_candidate(
        "def has_dupe(xs):\n    return len(set(xs)) != len(xs)\n",
        "has_dupe", [[[1, 2, 1]], [[1, 2]], [[]]],
    )
    assert result.ok, result
    assert result.status == "ok"
    assert result.outputs == [True, False, False]
