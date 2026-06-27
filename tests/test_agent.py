import json

import numpy as np

from zyn.agent import ToolAgent
from zyn.bpe import BPETokenizer
from zyn.tools import Sandbox


class ScriptedDecoder:
    def __init__(self, script, vocab_size):
        self.script = list(script)
        self.vocab_size = vocab_size
        self.i = 0

    def feed(self, ids):
        token = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        logits = np.full((1, 1, self.vocab_size), -1e9)
        logits[0, 0, token] = 1e9
        return logits


def _script_for_call(tok, call, answer):
    tc = tok.special_to_id["<tool_call>"]
    tce = tok.special_to_id["</tool_call>"]
    body = tok.encode(json.dumps(call, separators=(",", ":")))
    return [tc] + body + [tce] + tok.encode(answer) + [tok.eos_id]


def test_agent_executes_tool_and_resumes(tmp_path):
    tok = BPETokenizer()
    sandbox = Sandbox(tmp_path)
    call = {"name": "calculator", "arguments": {"expr": "2+2"}}
    script = _script_for_call(tok, call, "the answer is 4")
    decoder = ScriptedDecoder(script, tok.vocab_size)
    agent = ToolAgent(tok, sandbox, temperature=0.0)

    out = agent.run(decoder, [{"role": "user", "content": "what is 2+2"}])

    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0]["name"] == "calculator"
    assert out["tool_calls"][0]["result"] == "4"
    assert "the answer is 4" in out["answer"]


def test_agent_plain_answer_without_tool(tmp_path):
    tok = BPETokenizer()
    sandbox = Sandbox(tmp_path)
    script = tok.encode("just text") + [tok.eos_id]
    decoder = ScriptedDecoder(script, tok.vocab_size)
    agent = ToolAgent(tok, sandbox, temperature=0.0)

    out = agent.run(decoder, [{"role": "user", "content": "hi"}])

    assert out["tool_calls"] == []
    assert "just text" in out["answer"]


def test_agent_stops_at_max_turns(tmp_path):
    tok = BPETokenizer()
    sandbox = Sandbox(tmp_path)
    call = {"name": "calculator", "arguments": {"expr": "1+1"}}
    script = _script_for_call(tok, call, "done")
    decoder = ScriptedDecoder(script, tok.vocab_size)
    agent = ToolAgent(tok, sandbox, temperature=0.0, max_turns=1)

    out = agent.run(decoder, [{"role": "user", "content": "q"}])
    assert len(out["tool_calls"]) == 1


def test_agent_respects_max_new_tokens(tmp_path):
    tok = BPETokenizer()
    sandbox = Sandbox(tmp_path)
    script = tok.encode("a long answer that never ends")
    decoder = ScriptedDecoder(script, tok.vocab_size)
    agent = ToolAgent(tok, sandbox, temperature=0.0, max_new_tokens=5)

    out = agent.run(decoder, [{"role": "user", "content": "q"}])
    assert len(out["tokens"]) == 5
