from __future__ import annotations

import numpy as np

from zyn.bpe import BPETokenizer
from zyn.chat import render_messages
from zyn.tools import Sandbox, calculator, list_dir, write_file

_OPS = ["+", "-", "*"]
_CALC_TEMPLATES = [
    "What is {a} {op} {b}?",
    "Compute {a} {op} {b}.",
    "Please calculate {a} {op} {b}.",
]


def _calc_conversation(rng: np.random.Generator) -> list[dict]:
    a = int(rng.integers(2, 50))
    b = int(rng.integers(2, 50))
    op = str(rng.choice(_OPS))
    expr = f"{a} {op} {b}"
    result = calculator(None, expr)
    user = str(rng.choice(_CALC_TEMPLATES)).format(a=a, op=op, b=b)
    return [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "calculator", "arguments": {"expr": expr}}],
        },
        {"role": "tool", "content": result},
        {"role": "assistant", "content": f"The result is {result}."},
    ]


def _files_conversation(rng: np.random.Generator, sandbox: Sandbox) -> list[dict]:
    name = f"note_{int(rng.integers(0, 1000))}.txt"
    body = str(rng.choice(["hello world", "todo refactor", "print('hi')"]))
    convo_dir = sandbox.root / f"convo_{int(rng.integers(0, 1_000_000_000))}"
    convo_sandbox = Sandbox(
        convo_dir,
        allow_write=sandbox.allow_write,
        allow_shell=sandbox.allow_shell,
        timeout=sandbox.timeout,
    )
    write_file(convo_sandbox, name, body)
    result = list_dir(convo_sandbox, ".")
    return [
        {"role": "user", "content": "List the files here."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}],
        },
        {"role": "tool", "content": result},
        {"role": "assistant", "content": f"The directory contains:\n{result}"},
    ]


def generate_conversations(
    n: int,
    seed: int = 0,
    tools: tuple[str, ...] = ("calculator",),
    sandbox: Sandbox | None = None,
) -> list[list[dict]]:
    rng = np.random.default_rng(seed)
    conversations: list[list[dict]] = []
    for _ in range(n):
        choice = str(rng.choice(tools))
        if choice == "calculator":
            conversations.append(_calc_conversation(rng))
        elif choice == "list_dir":
            if sandbox is None:
                raise ValueError("list_dir conversations require a writable sandbox")
            conversations.append(_files_conversation(rng, sandbox))
        else:
            raise ValueError(f"unsupported synthetic tool {choice}")
    return conversations


def build_tool_stream(
    conversations: list[list[dict]],
    tok: BPETokenizer,
) -> tuple[np.ndarray, np.ndarray]:
    tokens: list[int] = []
    mask: list[int] = []
    for messages in conversations:
        ids, m = render_messages(messages, tok, add_bos=True)
        tokens.extend(ids)
        mask.extend(m)
    dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
    return np.asarray(tokens, dtype=dtype), np.asarray(mask, dtype=np.uint8)
