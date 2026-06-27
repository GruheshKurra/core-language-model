from __future__ import annotations

import ast
import operator
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class ToolError(Exception):
    pass


def _eval_arith(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_arith(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_arith(node.left)
        right = _eval_arith(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 64 or abs(left) > 1e6):
            raise ToolError("exponent out of range")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_arith(node.operand))
    raise ToolError("unsupported expression")


def calculator(sandbox: "Sandbox", expr: str) -> str:
    tree = ast.parse(expr, mode="eval")
    value = _eval_arith(tree)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class Sandbox:
    def __init__(
        self,
        root: str | Path,
        allow_write: bool = False,
        allow_shell: bool = False,
        timeout: float = 5.0,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allow_write = allow_write
        self.allow_shell = allow_shell
        self.timeout = timeout

    def resolve(self, rel: str) -> Path:
        target = (self.root / rel).resolve()
        root_str = str(self.root)
        if str(target) != root_str and not str(target).startswith(root_str + os.sep):
            raise ToolError("path escapes sandbox")
        return target


def read_file(sandbox: Sandbox, path: str) -> str:
    target = sandbox.resolve(path)
    if not target.is_file():
        raise ToolError(f"no such file: {path}")
    return target.read_text(encoding="utf-8", errors="replace")


def write_file(sandbox: Sandbox, path: str, content: str) -> str:
    if not sandbox.allow_write:
        raise ToolError("write disabled")
    target = sandbox.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    target.write_bytes(data)
    return f"wrote {len(data)} bytes to {path}"


def edit_file(sandbox: Sandbox, path: str, old: str, new: str) -> str:
    if not sandbox.allow_write:
        raise ToolError("write disabled")
    target = sandbox.resolve(path)
    if not target.is_file():
        raise ToolError(f"no such file: {path}")
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise ToolError("old text not found")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"edited {path}"


def list_dir(sandbox: Sandbox, path: str = ".") -> str:
    target = sandbox.resolve(path)
    if not target.is_dir():
        raise ToolError(f"no such directory: {path}")
    names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(names)


def glob_files(sandbox: Sandbox, pattern: str) -> str:
    root_str = str(sandbox.root)
    matches: list[str] = []
    for p in sandbox.root.glob(pattern):
        rp = str(p.resolve())
        if not rp.startswith(root_str + os.sep):
            continue
        matches.append(str(Path(rp).relative_to(sandbox.root)))
    return "\n".join(sorted(matches))


def grep(sandbox: Sandbox, pattern: str, path: str = ".", max_matches: int = 200) -> str:
    regex = re.compile(pattern)
    base = sandbox.resolve(path)
    files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
    out: list[str] = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        rel = f.relative_to(sandbox.root)
        for n, line in enumerate(lines, start=1):
            if regex.search(line):
                out.append(f"{rel}:{n}:{line}")
                if len(out) >= max_matches:
                    return "\n".join(out)
    return "\n".join(out)


def bash(sandbox: Sandbox, command: str) -> str:
    if not sandbox.allow_shell:
        raise ToolError("shell disabled")
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(sandbox.root),
        capture_output=True,
        text=True,
        timeout=sandbox.timeout,
    )
    return (proc.stdout + proc.stderr).strip()


def python_eval(sandbox: Sandbox, code: str) -> str:
    if not sandbox.allow_shell:
        raise ToolError("python execution disabled")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(sandbox.root),
        capture_output=True,
        text=True,
        timeout=sandbox.timeout,
    )
    return (proc.stdout + proc.stderr).strip()


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


TOOLS: dict[str, ToolSpec] = {
    "calculator": ToolSpec(
        "calculator",
        "Evaluate an arithmetic expression and return the numeric result.",
        _schema({"expr": {"type": "string"}}, ["expr"]),
        calculator,
    ),
    "read_file": ToolSpec(
        "read_file",
        "Read a UTF-8 text file inside the sandbox.",
        _schema({"path": {"type": "string"}}, ["path"]),
        read_file,
    ),
    "write_file": ToolSpec(
        "write_file",
        "Write text to a file inside the sandbox.",
        _schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        write_file,
    ),
    "edit_file": ToolSpec(
        "edit_file",
        "Replace the first occurrence of old text with new text in a file.",
        _schema(
            {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            ["path", "old", "new"],
        ),
        edit_file,
    ),
    "list_dir": ToolSpec(
        "list_dir",
        "List entries of a directory inside the sandbox.",
        _schema({"path": {"type": "string"}}, []),
        list_dir,
    ),
    "glob": ToolSpec(
        "glob",
        "Find files matching a glob pattern inside the sandbox.",
        _schema({"pattern": {"type": "string"}}, ["pattern"]),
        glob_files,
    ),
    "grep": ToolSpec(
        "grep",
        "Search file contents for a regular expression inside the sandbox.",
        _schema({"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
        grep,
    ),
    "bash": ToolSpec(
        "bash",
        "Run a shell command from the sandbox directory (gated; NOT isolated — full host filesystem and network access).",
        _schema({"command": {"type": "string"}}, ["command"]),
        bash,
    ),
    "python_eval": ToolSpec(
        "python_eval",
        "Run a short Python program from the sandbox directory (gated; NOT isolated — full host filesystem and network access).",
        _schema({"code": {"type": "string"}}, ["code"]),
        python_eval,
    ),
}


def tool_schemas() -> list[dict]:
    return [
        {"name": s.name, "description": s.description, "parameters": s.parameters}
        for s in TOOLS.values()
    ]


def execute(name: str, arguments: dict, sandbox: Sandbox) -> str:
    spec = TOOLS.get(name)
    if spec is None:
        return f"Error: unknown tool {name}"
    try:
        return spec.func(sandbox, **arguments)
    except ToolError as exc:
        return f"Error: {exc}"
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}"
    except subprocess.TimeoutExpired:
        return f"Error: {name} timed out"
    except Exception as exc:
        return f"Error: {name} failed: {exc}"
