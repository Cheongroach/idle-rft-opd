"""MBPP verifier: untrusted Python in bwrap, value comparison in the parent.

No eval/exec of candidate, serialized object, or assertion in the parent.
Inputs reach the guest; expected outputs and assertions never do. This is a
Linux CPU harness, not a claim that arbitrary GPU workers have isolation.
"""
import ast
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


class InfrastructureError(RuntimeError):
    pass


def pack(value):
    if type(value) in (type(None), bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float and math.isfinite(value):
        return ["float", value]
    if type(value) is complex:
        return ["complex", [pack(value.real), pack(value.imag)]]
    if type(value) in (list, tuple, set, frozenset):
        return [type(value).__name__, [pack(v) for v in value]]
    if type(value) is dict:
        return ["dict", [[pack(k), pack(v)] for k, v in value.items()]]
    raise ValueError("Unsupported result type")


def unpack(value, depth=0):
    if depth > 64 or type(value) is not list or len(value) != 2:
        raise ValueError("Invalid value protocol")
    tag, data = value
    primitive = {"NoneType": type(None), "bool": bool, "int": int,
                 "str": str, "float": float}
    if tag in primitive:
        if type(data) is not primitive[tag]:
            raise ValueError("Wrong primitive type")
        if tag == "float" and not math.isfinite(data):
            raise ValueError("Nonfinite output")
        return data
    if type(data) is not list:
        raise ValueError("Wrong container type")
    if tag == "complex" and len(data) == 2:
        return complex(*(unpack(v, depth + 1) for v in data))
    if tag in ("list", "tuple", "set", "frozenset"):
        constructors = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}
        return constructors[tag](unpack(v, depth + 1) for v in data)
    if tag == "dict":
        return {unpack(k, depth + 1): unpack(v, depth + 1) for k, v in data}
    raise ValueError("Unknown value tag")


GUEST = r'''
import ctypes, ctypes.util, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 3))
resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
request = json.loads(sys.stdin.read(131072))
lib = ctypes.CDLL("libseccomp.so.2")
lib.seccomp_init.argtypes = [ctypes.c_uint32]
lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
ctx = lib.seccomp_init(0x7fff0000)
if not ctx: raise RuntimeError("seccomp init failed")
for name in ("clone", "clone3", "fork", "vfork", "execve", "execveat", "ptrace",
             "mount", "umount2", "bpf", "unshare", "setns"):
    nr = lib.seccomp_syscall_resolve_name(name.encode())
    if nr >= 0 and lib.seccomp_rule_add(ctx, 0x00050001, nr, 0) != 0:
        raise RuntimeError("seccomp rule failed")
if lib.seccomp_load(ctx) != 0: raise RuntimeError("seccomp load failed")
def decode(v):
    tag, x = v
    if tag in ("NoneType","bool","int","str","float"): return x
    if tag == "complex": return complex(*[decode(a) for a in x])
    if tag == "dict": return {decode(k):decode(a) for k,a in x}
    return {"list":list,"tuple":tuple,"set":set,"frozenset":frozenset}[tag](decode(a) for a in x)
def encode(v, depth=0):
    if depth > 64: raise ValueError("output too deep")
    if type(v) in (type(None),bool,int,str,float): return [type(v).__name__,v]
    if type(v) is complex: return ["complex",[encode(v.real),encode(v.imag)]]
    if type(v) in (list,tuple,set,frozenset):
        if len(v)>10000: raise ValueError("output too large")
        return [type(v).__name__,[encode(a,depth+1) for a in v]]
    if type(v) is dict:
        if len(v)>10000: raise ValueError("output too large")
        return ["dict",[[encode(k,depth+1),encode(a,depth+1)] for k,a in v.items()]]
    raise ValueError("unsupported output type")
namespace = {"__name__":"candidate"}
exec(compile(request["code"],"<candidate>","exec"),namespace)
results=[]
for call in request["calls"]:
    value=namespace[call["name"]](*decode(call["args"]),**decode(call["kwargs"]))
    results.append(encode(value))
sys.stdout.write("\nIDLE_RFT_RESULT:"+json.dumps(results,allow_nan=False,separators=(",",":"))+"\n")
'''


def execute(code, calls, timeout=5):
    executable = shutil.which("bwrap")
    if not executable:
        raise InfrastructureError("bwrap unavailable; refusing unsafe fallback")
    payload = json.dumps({"code": code, "calls": [
        {"name": c["name"], "args": pack(c["args"]), "kwargs": pack(c["kwargs"])}
        for c in calls]}, allow_nan=False).encode()
    if len(payload) > 131000:
        return {"status": "invalid_program", "reason": "payload_limit"}
    command = [executable, "--unshare-all", "--die-with-parent", "--clearenv",
               "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib"]
    if Path("/lib64").exists():
        command += ["--ro-bind", "/lib64", "/lib64"]
    command += ["--dev", "/dev", "--tmpfs", "/tmp",
                "/usr/bin/python3", "-I", "-S", "-c", GUEST]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = subprocess.run(command, input=payload, stdout=stdout, stderr=stderr,
                                    timeout=timeout, close_fds=True)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        stdout.seek(0)
        stderr.seek(0)
        output, error = stdout.read(65537), stderr.read(65537)
    if b"bwrap:" in error or b"seccomp" in error:
        raise InfrastructureError(error[:500].decode(errors="replace"))
    if result.returncode:
        return {"status": "runtime_error", "returncode": result.returncode}
    if len(output) > 65536:
        return {"status": "invalid_program", "reason": "output_limit"}
    try:
        marker = b"\nIDLE_RFT_RESULT:"
        # A candidate may forge bytes but cannot inject custom objects or equality.
        raw = json.loads(output.rsplit(marker, 1)[1])
        if type(raw) is not list or len(raw) != len(calls):
            raise ValueError("Wrong output count")
        values = [unpack(v) for v in raw]
        return {"status": "ok", "values": values}
    except (ValueError, TypeError, IndexError, KeyError, RecursionError):
        return {"status": "invalid_program", "reason": "output_protocol"}


TRANSFORMS = {"set": set, "list": list, "tuple": tuple, "sorted": sorted,
              "len": len, "round": round, "abs": abs}


def assertions(tests):
    expressions = []
    for test in tests:
        tree = ast.parse(test)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
            raise ValueError("Only simple assertions supported")
        expressions.append(tree.body[0].test)
    return expressions


def interpret(node, call):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        constructor = {ast.Tuple: tuple, ast.List: list, ast.Set: set}[type(node)]
        return constructor(interpret(v, call) for v in node.elts)
    if isinstance(node, ast.Dict):
        return {interpret(k, call): interpret(v, call) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp):
        value = interpret(node.operand, call)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.Not):
            return not value
    if isinstance(node, ast.BinOp):
        a, b = interpret(node.left, call), interpret(node.right, call)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Mult):
            return a * b
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        return interpret(node.left, call) == interpret(node.comparators[0], call)
    if isinstance(node, ast.Call):
        args = [interpret(a, call) for a in node.args]
        kwargs = {k.arg: interpret(k.value, call) for k in node.keywords}
        if isinstance(node.func, ast.Name):
            if node.func.id in TRANSFORMS:
                return TRANSFORMS[node.func.id](*args, **kwargs)
            return call(node.func.id, args, kwargs)
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math" and node.func.attr == "isclose"):
            return math.isclose(*args, **kwargs)
    raise ValueError(f"Unsupported assertion node: {type(node).__name__}")


def test_calls(tests):
    calls = []
    def collect(name, args, kwargs):
        calls.append({"name": name, "args": args, "kwargs": kwargs})
        # Permit transforms around placeholder outputs when validating the AST.
        return 0
    for expression in assertions(tests):
        # Collect syntactically instead of evaluating transformed placeholders.
        for node in ast.walk(expression):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id not in TRANSFORMS):
                collect(node.func.id, [ast.literal_eval(a) for a in node.args],
                        {k.arg: ast.literal_eval(k.value) for k in node.keywords})
    if not calls:
        raise ValueError("No candidate function calls")
    return calls


def verify(code, tests):
    try:
        calls = test_calls(tests)
        ast.parse(code)
    except (SyntaxError, ValueError):
        return {"status": "syntax_or_contract_error", "passed": False}
    result = execute(code, calls)
    if result["status"] != "ok":
        return {**result, "passed": False}
    outputs = iter(result["values"])
    try:
        matches = [bool(interpret(e, lambda *_: next(outputs))) for e in assertions(tests)]
    except (ValueError, TypeError, StopIteration, OverflowError):
        return {"status": "unsupported_comparison", "passed": False}
    return {"status": "pass" if all(matches) else "wrong_answer",
            "passed": all(matches), "tests": len(matches), "matches": matches}


def mutations(code, limit=2):
    """Basic test-killing audit; survivors are not automatically real defects."""
    tree = ast.parse(code)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            old = node.value
            node.value = ast.Constant(None)
            yield ast.unparse(ast.fix_missing_locations(tree))
            node.value = old
            count += 1
        elif isinstance(node, ast.Compare) and node.ops:
            replacements = {ast.Lt: ast.LtE, ast.LtE: ast.Lt,
                            ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq}
            if type(node.ops[0]) in replacements:
                old = node.ops[0]
                node.ops[0] = replacements[type(old)]()
                yield ast.unparse(tree)
                node.ops[0] = old
                count += 1
        if count >= limit:
            return
