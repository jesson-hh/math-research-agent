import json
from unittest.mock import MagicMock

import pytest

from paper_distiller.llm.openai_compatible import (
    LLMClient,
    LLMError,
    ToolCall,
    ToolCallResponse,
)


def _fake_response(content):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def test_complete_basic(mocker):
    mock_post = mocker.patch("paper_distiller.llm.openai_compatible.httpx.Client.post")
    mock_post.return_value.json.return_value = _fake_response("hello back")
    mock_post.return_value.raise_for_status = MagicMock()

    client = LLMClient(api_key="sk-test", base_url="https://x.test/v1", model="qwen-plus")
    out = client.complete([{"role": "user", "content": "hi"}])
    assert out == "hello back"


def test_complete_tracks_token_usage(mocker):
    mock_post = mocker.patch("paper_distiller.llm.openai_compatible.httpx.Client.post")
    mock_post.return_value.json.return_value = _fake_response("response")
    mock_post.return_value.raise_for_status = MagicMock()

    client = LLMClient(api_key="sk-test", base_url="https://x.test/v1", model="qwen-plus")
    client.complete([{"role": "user", "content": "hi"}])
    assert client.total_tokens_in == 100
    assert client.total_tokens_out == 50

    client.complete([{"role": "user", "content": "hi again"}])
    assert client.total_tokens_in == 200
    assert client.total_tokens_out == 100


def test_complete_response_format_json(mocker):
    mock_post = mocker.patch("paper_distiller.llm.openai_compatible.httpx.Client.post")
    mock_post.return_value.json.return_value = _fake_response('{"x": 1}')
    mock_post.return_value.raise_for_status = MagicMock()

    client = LLMClient(api_key="sk-test", base_url="https://x.test/v1", model="qwen-plus")
    client.complete([{"role": "user", "content": "hi"}], response_format="json")

    call_kwargs = mock_post.call_args.kwargs
    body = call_kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}


def _fake_tool_response(json_body):
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    resp.text = json.dumps(json_body)
    return resp


def test_complete_with_tools_returns_text_only(mocker):
    """When LLM doesn't invoke tools, text is filled and tool_calls is empty."""
    fake_resp = _fake_tool_response({
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello world"},
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10},
    })
    mocker.patch(
        "paper_distiller.llm.openai_compatible.httpx.Client.post",
        return_value=fake_resp,
    )
    llm = LLMClient(api_key="sk-test", base_url="https://x.test/v1", model="qwen-plus")
    out = llm.complete_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{
            "type": "function",
            "function": {"name": "search", "description": "...", "parameters": {}},
        }],
    )
    assert isinstance(out, ToolCallResponse)
    assert out.text == "Hello world"
    assert out.tool_calls == []
    assert out.finish_reason == "stop"
    assert llm.total_tokens_in == 50
    assert llm.total_tokens_out == 10


def test_complete_with_tools_returns_tool_call(mocker):
    """When LLM invokes a tool, tool_calls is populated."""
    fake_resp = _fake_tool_response({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"topic":"diffusion","n":5}',
                    },
                }],
            },
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
    })
    mocker.patch(
        "paper_distiller.llm.openai_compatible.httpx.Client.post",
        return_value=fake_resp,
    )
    llm = LLMClient(api_key="sk-test", base_url="https://x.test/v1", model="qwen-plus")
    out = llm.complete_with_tools(
        messages=[{"role": "user", "content": "find papers"}],
        tools=[],
    )
    assert out.text == ""
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "call_abc123"
    assert tc.name == "search"
    assert tc.arguments == {"topic": "diffusion", "n": 5}
    assert out.finish_reason == "tool_calls"


def test_complete_with_tools_handles_malformed_arguments(mocker):
    """If tool_call.arguments isn't valid JSON, default to empty dict (don't crash)."""
    fake_resp = _fake_tool_response({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_x",
                    "function": {"name": "search", "arguments": "not valid json"},
                }],
            },
        }],
        "usage": {"prompt_tokens": 80, "completion_tokens": 20},
    })
    mocker.patch(
        "paper_distiller.llm.openai_compatible.httpx.Client.post",
        return_value=fake_resp,
    )
    llm = LLMClient(api_key="sk-test", base_url="https://x.test/v1", model="qwen-plus")
    out = llm.complete_with_tools(messages=[{"role": "user", "content": "x"}], tools=[])
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].arguments == {}
