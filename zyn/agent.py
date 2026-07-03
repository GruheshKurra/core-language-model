from __future__ import annotations

import numpy as np

from zyn.bpe import BPETokenizer
from zyn.chat import build_prompt, parse_tool_call
from zyn.generate import _sample_next
from zyn.tools import Sandbox, execute


class ToolAgent:
    def __init__(
        self,
        tok: BPETokenizer,
        sandbox: Sandbox,
        max_new_tokens: int = 128,
        max_turns: int = 4,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.tok = tok
        self.sandbox = sandbox
        self.max_new_tokens = max_new_tokens
        self.max_turns = max_turns
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rng = rng if rng is not None else np.random.default_rng()

    def run(self, decoder, messages: list[dict]) -> dict:
        tok = self.tok
        tc = tok.special_to_id["<tool_call>"]
        tce = tok.special_to_id["</tool_call>"]
        tr = tok.special_to_id["<tool_result>"]

        prompt = build_prompt(messages, tok)
        logits = decoder.feed(np.asarray(prompt)[None, :])[:, -1, :]

        generated: list[int] = []
        answer_ids: list[int] = []
        collect: list[int] = []
        tool_calls: list[dict] = []
        collecting = False
        turns = 0

        for _ in range(self.max_new_tokens):
            nid = int(_sample_next(logits, self.temperature, self.top_k, self.top_p, self.rng)[0])
            generated.append(nid)

            if nid == tok.eos_id:
                break

            if nid == tc:
                collecting = True
                collect = []
                logits = decoder.feed(np.asarray([[nid]]))[:, -1, :]
                continue

            if nid == tce:
                collecting = False
                call = parse_tool_call(tok.decode(collect, skip_specials=True))
                result = execute(call["name"], call["arguments"], self.sandbox)
                tool_calls.append({"name": call["name"], "arguments": call["arguments"], "result": result})
                inject = [tce, tr] + tok.encode(result + "\n")
                logits = decoder.feed(np.asarray([inject]))[:, -1, :]
                turns += 1
                if turns >= self.max_turns:
                    break
                continue

            if collecting:
                collect.append(nid)
            else:
                answer_ids.append(nid)
            logits = decoder.feed(np.asarray([[nid]]))[:, -1, :]

        return {
            "text": tok.decode(generated, skip_specials=True),
            "answer": tok.decode(answer_ids, skip_specials=True),
            "tool_calls": tool_calls,
            "tokens": generated,
        }
