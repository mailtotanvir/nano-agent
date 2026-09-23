"""Namespace-isolated runner. Never execute candidate code in this process."""
import json
import os
import selectors
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

BWRAP = "/usr/bin/bwrap"


@dataclass(frozen=True)
class CachegrindConfig:
    """Trusted local tool paths; never candidate-controlled.

    tool_memory_mb caps the whole profiler's kernel address space (default 1 GiB).
    run_candidate.memory_mb separately caps the emulated Python client's address
    space. Both limits are enforced; no memory limit is silently disabled.
    Profiling additionally requires /usr/bin/setarch and /usr/bin/prlimit.
    """
    binary: Path
    lib_dir: Path | None = None
    tool_memory_mb: int = 1024


@dataclass
class SandboxResult:
    ok: bool
    outputs: list = field(default_factory=list)
    status: str = "ok"
    detail: str = ""
    profile: str | None = None


def isolation_command() -> list[str]:
    """Trusted bootstrap prefix, NOT an alternative candidate execution API.

    Append mounts then ``--`` and a trusted bootstrap installing resource limits.
    Never append raw candidate code: use run_candidate for untrusted programs.
    """
    command = [BWRAP, "--unshare-all", "--die-with-parent", "--new-session",
               "--as-pid-1", "--proc", "/proc", "--cap-drop", "ALL", "--ro-bind", "/usr", "/usr",
               "--ro-bind", "/lib", "/lib"]
    # Debian-derived verifier containers commonly omit /lib64; binding a
    # non-existent source makes Bubblewrap fail before untrusted code runs.
    if Path("/lib64").is_dir():
        command += ["--ro-bind", "/lib64", "/lib64"]
    if Path("/usr/local").is_dir():
        command += ["--ro-bind", "/usr/local", "/usr/local"]
    return command + ["--tmpfs", "/tmp", "--chdir", "/tmp", "--clearenv",
                      "--setenv", "PYTHONHASHSEED", "0", "--setenv", "LC_ALL", "C",
                      "--setenv", "PYTHONMALLOC", "malloc", "--setenv", "PATH", "/usr/bin:/bin"]


def _capture(command, payload, timeout_s, max_output_bytes):
    """Bound both pipes together; kill the namespace on timeout/overflow."""
    with tempfile.TemporaryFile() as request:
        request.write(payload.encode())
        request.seek(0)
        with subprocess.Popen(command, stdin=request, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env={}, start_new_session=True) as child:
            try:
                deadline = time.monotonic() + timeout_s
                chunks = {"stdout": bytearray(), "stderr": bytearray()}
                size = 0
                with selectors.DefaultSelector() as selector:
                    selector.register(child.stdout, selectors.EVENT_READ, "stdout")
                    selector.register(child.stderr, selectors.EVENT_READ, "stderr")
                    while selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return "timeout", "wall-clock limit", None
                        for key, _ in selector.select(remaining):
                            chunk = os.read(key.fileobj.fileno(), 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            size += len(chunk)
                            if size > max_output_bytes:
                                return "output_limit", "combined stdout/stderr limit", None
                            chunks[key.data].extend(chunk)
                child.wait(timeout=max(0.001, deadline - time.monotonic()))
                return "ok", chunks, child.returncode
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def run_candidate(source: str, function_name: str, inputs: list[list], *,
                  timeout_s: float = 2, memory_mb: int = 128, cpu_seconds: int = 1,
                  max_output_bytes: int = 1048576,
                  cachegrind: CachegrindConfig | None = None) -> SandboxResult:
    """Run one restricted pure function on positional JSON argument lists.

    Defaults: 2 s wall, 1 s CPU, 128 MiB address space, 1 MiB combined output.
    Requests are capped at 1 MiB JSON; source at 65536 characters. Results must
    be strict finite JSON, depth <=64 including the outer outputs list. Imports,
    private attributes, arbitrary calls, defaults/decorators and nested functions
    are denied. There is no print/open/eval/exec/import builtin in the candidate.

    The OS boundary is bwrap namespaces + read-only /usr, /lib, /lib64, private
    proc/tmp and no host home/etc/repository. Missing isolation fails closed.
    This is not a VM or protection against kernel/interpreter vulnerabilities.
    The AST/builtin policy also protects same-process profiler files: there is
    no separate UID between Python and Valgrind. Local tool paths are trusted.
    profile is raw Cachegrind text only after successful execution; consumers
    must still validate its summary and independently check repeatability.
    """
    if (not isinstance(source, str) or len(source) > 65536
            or not isinstance(function_name, str) or not function_name.isidentifier()
            or type(inputs) is not list or any(type(args) is not list for args in inputs)
            or type(timeout_s) not in (int, float) or not 0 < timeout_s <= 3600
            or type(memory_mb) is not int or not 32 <= memory_mb <= 4096
            or type(cpu_seconds) is not int or not 1 <= cpu_seconds <= 3600
            or type(max_output_bytes) is not int or not 128 <= max_output_bytes <= 16777216):
        return SandboxResult(False, status="invalid_request", detail="invalid shape or resource limits")
    try:
        request = json.dumps({"source": source, "function_name": function_name, "inputs": inputs,
                              "cpu_seconds": cpu_seconds, "memory_mb": memory_mb}, allow_nan=False)
        if len(request) > 1048576:
            raise ValueError("request exceeds 1 MiB")
    except (ValueError, TypeError, RecursionError) as exc:
        return SandboxResult(False, status="invalid_request", detail=type(exc).__name__)
    worker = Path(__file__).with_name("worker.py").read_text()
    command = isolation_command()
    # The isolation boundary mounts /usr, not the caller's virtualenv.
    # Pin the same system interpreter used by the verifier image.
    interpreter = ["/usr/bin/python3", "-s", "-S", "-c",
                   "import sys; exec(sys.stdin.readline())"]
    payload = "exec(" + repr(worker) + ")\n" + request
    try:
        with tempfile.TemporaryDirectory(prefix="speedup-") as directory:
            if cachegrind is not None:
                binary = Path(cachegrind.binary).resolve(strict=True)
                command += ["--symlink", "usr/bin", "/bin", "--ro-bind", str(binary), "/tools/valgrind"]
                companion = binary.with_name(binary.name + ".bin")
                if companion.is_file():
                    command += ["--ro-bind", str(companion), "/tools/valgrind.bin"]
                if cachegrind.lib_dir is not None:
                    library = Path(cachegrind.lib_dir).resolve(strict=True)
                    command += ["--ro-bind", str(library), "/tools/lib",
                                "--setenv", "VALGRIND_LIB", "/tools/lib"]
                command += ["--bind", directory, "/work"]
                interpreter = ["/usr/bin/setarch", os.uname().machine, "-R", "/tools/valgrind",
                               "--tool=cachegrind", "--cachegrind-out-file=/work/cachegrind.out",
                               "--"] + interpreter
                interpreter = ["/usr/bin/prlimit",
                               f"--as={cachegrind.tool_memory_mb * 1024 * 1024}",
                               f"--cpu={cpu_seconds}", "--fsize=8388608", "--core=0",
                               "--"] + interpreter
            command += ["--"] + interpreter
            status, output, returncode = _capture(command, payload, timeout_s, max_output_bytes)
            if status != "ok":
                return SandboxResult(False, status=status, detail=output)
            if returncode in (-signal.SIGKILL, -signal.SIGXCPU, -signal.SIGXFSZ,
                              128 + signal.SIGKILL, 128 + signal.SIGXCPU, 128 + signal.SIGXFSZ):
                return SandboxResult(False, status="resource_limit", detail="worker killed by resource signal")
            if returncode:
                return SandboxResult(False, status="sandbox_error",
                                     detail=output["stderr"][:512].decode(errors="replace"))
            result = SandboxResult(**json.loads(output["stdout"]))
            if result.ok and cachegrind is not None:
                with open(Path(directory) / "cachegrind.out", "r") as profile:
                    result.profile = profile.read(8388609)
                if len(result.profile) > 8388608:
                    return SandboxResult(False, status="output_limit", detail="profile limit")
            return result
    except subprocess.TimeoutExpired:
        return SandboxResult(False, status="timeout", detail="wall-clock limit")
    except (OSError, ValueError, TypeError) as exc:
        return SandboxResult(False, status="sandbox_error", detail=type(exc).__name__)
