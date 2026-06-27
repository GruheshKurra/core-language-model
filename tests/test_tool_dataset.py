import numpy as np

from zyn.bpe import BPETokenizer
from zyn.chat import parse_tool_call
from zyn.tooldata import build_tool_stream, generate_conversations
from zyn.tools import Sandbox, execute


def test_calc_traces_roundtrip(tmp_path):
    convs = generate_conversations(20, seed=1, tools=("calculator",))
    sandbox = Sandbox(tmp_path)
    for messages in convs:
        assistant = messages[1]
        tool_msg = messages[2]
        call = assistant["tool_calls"][0]
        rerun = execute(call["name"], call["arguments"], sandbox)
        assert rerun == tool_msg["content"]


def test_parse_call_from_rendered_specials():
    convs = generate_conversations(1, seed=2, tools=("calculator",))
    call = convs[0][1]["tool_calls"][0]
    import json

    parsed = parse_tool_call(json.dumps(call))
    assert parsed["name"] == "calculator"


def test_stream_has_tool_specials_and_mask():
    tok = BPETokenizer()
    convs = generate_conversations(10, seed=3, tools=("calculator",))
    tokens, mask = build_tool_stream(convs, tok)
    assert tokens.shape == mask.shape
    assert tok.special_to_id["<tool_call>"] in tokens.tolist()
    assert tok.special_to_id["</tool_call>"] in tokens.tolist()
    assert tok.special_to_id["<tool_result>"] in tokens.tolist()
    assert int(mask.max()) == 1


def test_tool_result_tokens_not_trained():
    tok = BPETokenizer()
    convs = generate_conversations(5, seed=4, tools=("calculator",))
    tokens, mask = build_tool_stream(convs, tok)
    tr = tok.special_to_id["<tool_result>"]
    positions = np.where(tokens == tr)[0]
    assert len(positions) > 0
    for p in positions:
        assert mask[p] == 0
