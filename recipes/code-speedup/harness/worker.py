"""Trusted bootstrap sent over stdin; no harness/oracle source is mounted."""
import ast
import json
import math
import os
import resource
import sys


class PolicyError(ValueError):
    pass


_ALLOWED_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AugAssign, ast.Expr, ast.For, ast.While, ast.If, ast.IfExp, ast.Break,
    ast.Continue, ast.Pass, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Subscript, ast.Slice,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Attribute,
    ast.keyword, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.comprehension, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.UAdd, ast.USub, ast.Not, ast.Invert, ast.And, ast.Or,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
}
_SAFE_BUILTINS = {"len": len, "set": set, "range": range, "abs": abs, "sorted": sorted,
                  "sum": sum, "min": min, "max": max, "all": all, "any": any,
                  "enumerate": enumerate, "zip": zip, "reversed": reversed,
                  "list": list, "tuple": tuple, "dict": dict, "frozenset": frozenset,
                  "int": int, "float": float, "bool": bool, "str": str,
                  "round": round, "divmod": divmod, "pow": pow, "ord": ord, "chr": chr}
_SAFE_METHODS = {"append", "extend", "insert", "pop", "remove", "clear", "copy",
                 "count", "index", "reverse", "sort", "get", "keys", "values",
                 "items", "update", "setdefault", "add", "discard", "union",
                 "intersection", "difference", "issubset", "issuperset",
                 "join", "split", "strip", "lower", "upper", "replace",
                 "startswith", "endswith", "find", "isdigit", "isalpha"}


def validate(source, function_name):
    tree = ast.parse(source)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != function_name):
        raise PolicyError("exactly one named function required")
    function = tree.body[0]
    if (function.decorator_list or function.args.defaults
            or any(value is not None for value in function.args.kw_defaults)):
        raise PolicyError("decorators and defaults are forbidden")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise PolicyError("forbidden syntax: " + type(node).__name__)
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise PolicyError("nested functions are forbidden")
        name = (node.id if isinstance(node, ast.Name) else
                node.arg if isinstance(node, ast.arg) else
                node.name if isinstance(node, ast.FunctionDef) else "")
        if name.startswith("_"):
            raise PolicyError("private names are forbidden")
        if isinstance(node, ast.Attribute) and node.attr not in _SAFE_METHODS:
            raise PolicyError("forbidden attribute")
        if isinstance(node, ast.Call):
            target = node.func
            if not (isinstance(target, ast.Name) and target.id in {*_SAFE_BUILTINS, function_name}
                    or isinstance(target, ast.Attribute) and target.attr in _SAFE_METHODS):
                raise PolicyError("forbidden call")
    return compile(tree, "<candidate>", "exec")


def valid_output(value, depth=0):
    """Reject lossy JSON coercion, cycles and deep outputs before serialization."""
    if depth > 64:
        return False
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(valid_output(item, depth + 1) for item in value)
    if type(value) is dict:
        return all(type(key) is str and valid_output(item, depth + 1)
                   for key, item in value.items())
    return False


def main():
    request = json.load(sys.stdin)
    # Neither /proc/1/cmdline nor /proc/1/fd/0 retains bootstrap/oracle data.
    sys.stdin.close()
    try:
        os.close(0)
    except OSError:
        pass
    cpu = request["cpu_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    memory = request["memory_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
    namespace = {"__builtins__": _SAFE_BUILTINS.copy()}
    try:
        # This process already has namespaces, rlimits, and a restricted AST.
        exec(validate(request["source"], request["function_name"]), namespace)  # noqa: S102
        outputs = [namespace[request["function_name"]](*args) for args in request["inputs"]]
        result = {"ok": True, "outputs": outputs, "status": "ok", "detail": ""}
    except (PolicyError, SyntaxError) as exc:
        result = {"ok": False, "outputs": [], "status": "policy_error", "detail": str(exc)[:256]}
    except MemoryError:
        result = {"ok": False, "outputs": [], "status": "memory_limit", "detail": "address-space limit"}
    except Exception as exc:  # noqa: BLE001 - contain arbitrary candidate failures inside this worker
        result = {"ok": False, "outputs": [], "status": "runtime_error", "detail": type(exc).__name__}
    try:
        if not valid_output(result["outputs"]):
            raise ValueError("result must be strict JSON of depth <= 64")
        encoded = json.dumps(result, allow_nan=False)
    except MemoryError:
        encoded = json.dumps({"ok": False, "outputs": [], "status": "memory_limit", "detail": "serialization limit"})
    except (ValueError, TypeError, RecursionError):
        encoded = json.dumps({"ok": False, "outputs": [], "status": "output_error", "detail": "non-JSON result"})
    print(encoded)


if __name__ == "__main__":
    main()
