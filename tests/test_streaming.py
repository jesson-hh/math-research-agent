"""Tests for LLMClient.complete_with_tools_stream — SSE chunk accumulation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock


def _sse_lines(payloads):
    """Build the byte-string list mimicking httpx's iter_lines() over SSE."""
    out = []
    for p in payloads:
        out.append(f"data: {json.dumps(p)}")
        out.append("")
    out.append("data: [DONE]")
    return out


def test_stream_yields_text_deltas(mocker):
    from paper_distiller.llm.openai_compatible import LLMClient, StreamChunk

    client = LLMClient("k", "https://x/v1", "qwen-plus")

    fake_stream_resp = MagicMock()
    fake_stream_resp.__enter__.return_value = fake_stream_resp
    fake_stream_resp.__exit__.return_value = False
    fake_stream_resp.iter_lines.return_value = iter(_sse_lines([
        {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "world"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]))
    fake_stream_resp.raise_for_status = MagicMock()

    mocker.patch.object(client._client, "stream", return_value=fake_stream_resp)

    chunks = list(client.complete_with_tools_stream(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "x"}}],
    ))

    assert isinstance(chunks[0], StreamChunk)
    text = "".join(c.text_delta for c in chunks)
    assert text == "Hello world"
    assert chunks[-1].finish_reason == "stop"


def test_stream_accumulates_tool_call(mocker):
    from paper_distiller.llm.openai_compatible import LLMClient, StreamChunk

    client = LLMClient("k", "https://x/v1", "qwen-plus")

    fake_resp = MagicMock()
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = False
    fake_resp.raise_for_status = MagicMock()
    fake_resp.iter_lines.return_value = iter(_sse_lines([
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call-abc",
            "function": {"name": "search", "arguments": ""},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"arguments": '{"topic":'},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"arguments": '"x"}'},
        }]}, "finish_reason": "tool_calls"}]},
    ]))

    mocker.patch.object(client._client, "stream", return_value=fake_resp)

    chunks = list(client.complete_with_tools_stream(
        messages=[{"role": "user", "content": "search"}],
        tools=[{"type": "function", "function": {"name": "search"}}],
    ))

    name_pieces = [c.tool_name_delta for c in chunks if c.tool_name_delta]
    arg_pieces = [c.tool_arg_delta for c in chunks if c.tool_arg_delta]
    assert "".join(name_pieces) == "search"
    assert "".join(arg_pieces) == '{"topic":"x"}'
    call_ids = [c.tool_call_id for c in chunks if c.tool_call_id]
    assert call_ids[0] == "call-abc"
    assert chunks[-1].finish_reason == "tool_calls"


def test_stream_updates_token_counts_from_usage(mocker):
    from paper_distiller.llm.openai_compatible import LLMClient

    client = LLMClient("k", "https://x/v1", "qwen-plus")
    fake_resp = MagicMock()
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = False
    fake_resp.raise_for_status = MagicMock()
    fake_resp.iter_lines.return_value = iter(_sse_lines([
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 50}},
    ]))

    mocker.patch.object(client._client, "stream", return_value=fake_resp)
    list(client.complete_with_tools_stream(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
    ))
    assert client.total_tokens_in == 100
    assert client.total_tokens_out == 50


def test_stream_chunk_dataclass_defaults():
    from paper_distiller.llm.openai_compatible import StreamChunk
    c = StreamChunk()
    assert c.text_delta == ""
    assert c.tool_call_id is None
    assert c.tool_name_delta == ""
    assert c.tool_arg_delta == ""
    assert c.finish_reason == ""
