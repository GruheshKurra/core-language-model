from __future__ import annotations

import json

from zyn.bpe import BPETokenizer

SYSTEM_HEADER = "<|system|>\n"
USER_HEADER = "<|user|>\n"
ASSISTANT_HEADER = "<|assistant|>\n"
TURN_END = "\n"


def _call_to_json(call: dict) -> str:
    return json.dumps(
        {"name": call["name"], "arguments": call.get("arguments", {})},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def render_messages(
    messages: list[dict],
    tok: BPETokenizer,
    add_bos: bool = True,
) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []
    tc = tok.special_to_id["<tool_call>"]
    tce = tok.special_to_id["</tool_call>"]
    tr = tok.special_to_id["<tool_result>"]

    def add(seg: list[int], trainable: bool) -> None:
        ids.extend(seg)
        flag = 1 if trainable else 0
        mask.extend([flag] * len(seg))

    if add_bos:
        add([tok.bos_id], False)

    prev_role: str | None = None
    for m in messages:
        role = m["role"]
        if role == "system":
            add(tok.encode(SYSTEM_HEADER + m["content"] + TURN_END), False)
        elif role == "user":
            add(tok.encode(USER_HEADER + m["content"] + TURN_END), False)
        elif role == "tool":
            add([tr], False)
            add(tok.encode(m["content"] + TURN_END), False)
        elif role == "assistant":
            if prev_role != "tool":
                add(tok.encode(ASSISTANT_HEADER), False)
            content = m.get("content", "")
            if content:
                add(tok.encode(content), True)
            calls = m.get("tool_calls")
            if calls:
                for call in calls:
                    add([tc], True)
                    add(tok.encode(_call_to_json(call)), True)
                    add([tce], True)
            else:
                add([tok.eos_id], True)
        else:
            raise ValueError(f"unknown role {role}")
        prev_role = role

    return ids, mask


def build_prompt(messages: list[dict], tok: BPETokenizer) -> list[int]:
    ids, _ = render_messages(messages, tok, add_bos=True)
    last_role = messages[-1]["role"] if messages else None
    if last_role not in ("assistant", "tool"):
        ids = ids + tok.encode(ASSISTANT_HEADER)
    return ids


def parse_tool_call(text: str) -> dict:
    obj = json.loads(text)
    if "name" not in obj:
        raise ValueError("tool call missing 'name'")
    return {"name": obj["name"], "arguments": obj.get("arguments", {})}
