import json

from zyn.bpe import BPETokenizer
from zyn.chat import build_prompt, parse_tool_call, render_messages


def _tok():
    return BPETokenizer()


def test_starts_with_bos_and_masks_user():
    tok = _tok()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    ids, mask = render_messages(messages, tok)
    assert ids[0] == tok.bos_id
    assert mask[0] == 0
    assert len(ids) == len(mask)
    assert ids[-1] == tok.eos_id
    assert mask[-1] == 1


def test_assistant_content_is_trained_user_is_not():
    tok = _tok()
    messages = [
        {"role": "user", "content": "abc"},
        {"role": "assistant", "content": "xyz"},
    ]
    ids, mask = render_messages(messages, tok)
    trained = [i for i, m in zip(ids, mask) if m == 1]
    decoded = tok.decode(trained, skip_specials=True)
    assert "xyz" in decoded
    assert "abc" not in decoded


def test_tool_call_emits_specials_and_no_eos():
    tok = _tok()
    messages = [
        {"role": "user", "content": "add"},
        {
            "role": "assistant",
            "tool_calls": [{"name": "calculator", "arguments": {"expr": "2+2"}}],
        },
    ]
    ids, mask = render_messages(messages, tok)
    assert tok.special_to_id["<tool_call>"] in ids
    assert tok.special_to_id["</tool_call>"] in ids
    assert tok.eos_id not in ids[1:]


def test_tool_continuation_has_single_assistant_header():
    tok = _tok()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "tool_calls": [{"name": "calculator", "arguments": {"expr": "1+1"}}],
        },
        {"role": "tool", "content": "2"},
        {"role": "assistant", "content": "the answer is 2"},
    ]
    ids, mask = render_messages(messages, tok)
    full = tok.decode(ids, skip_specials=True)
    assert full.count("<|assistant|>") == 1
    tr_id = tok.special_to_id["<tool_result>"]
    tr_pos = ids.index(tr_id)
    assert mask[tr_pos] == 0


def test_tool_result_not_trained():
    tok = _tok()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "tool_calls": [{"name": "calculator", "arguments": {"expr": "1+1"}}],
        },
        {"role": "tool", "content": "2"},
        {"role": "assistant", "content": "done"},
    ]
    ids, mask = render_messages(messages, tok)
    trained = tok.decode([i for i, m in zip(ids, mask) if m == 1], skip_specials=True)
    assert "done" in trained


def test_build_prompt_appends_assistant_header():
    tok = _tok()
    messages = [{"role": "user", "content": "hi"}]
    ids = build_prompt(messages, tok)
    assert tok.decode(ids, skip_specials=True).endswith("<|assistant|>\n")


def test_parse_tool_call_roundtrip():
    call = {"name": "calculator", "arguments": {"expr": "3*4"}}
    text = json.dumps(call, separators=(",", ":"))
    parsed = parse_tool_call(text)
    assert parsed["name"] == "calculator"
    assert parsed["arguments"]["expr"] == "3*4"
