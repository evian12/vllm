# SPDX-License-Identifier: Apache-2.0
"""CPU-only HTTP contract checks for text completion routes."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.chat_completion import api_router as chat_router
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
from vllm.entrypoints.openai.completion import api_router as completion_router
from vllm.entrypoints.openai.completion.protocol import CompletionResponse
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo


class ErrorFactory:
    def __init__(self, code):
        self.code = code

    def create_error_response(self, message):
        return ErrorResponse(error=dict(message=str(message), type="TestError",
                                        param=None, code=self.code))


class Handler(ErrorFactory):
    def __init__(self, mode, code):
        super().__init__(code)
        self.mode = mode

    async def reply(self, *, chat=False, render=False):
        if self.mode == "raise":
            raise RuntimeError("simulated handler failure")
        if self.mode == "returned_error":
            return self.create_error_response("simulated returned error")
        if render:
            return [{"prompt_token_ids": [1, 2]}]
        if self.mode == "stream":
            async def events():
                yield 'data: {"choices": []}\n\n'
                yield 'data: [DONE]\n\n'
            return events()
        cls = ChatCompletionResponse if chat else CompletionResponse
        return cls(id="test-id", created=0, model="target", choices=[],
                   usage=UsageInfo(prompt_tokens=2, completion_tokens=1, total_tokens=3))

    async def create_completion(self, *args):
        return await self.reply()

    async def render_completion_request(self, *args):
        return await self.reply(render=True)

    async def create_chat_completion(self, *args):
        return await self.reply(chat=True)

    async def render_chat_request(self, *args):
        return await self.reply(chat=True, render=True)


def app_for(mode, code):
    app = FastAPI()
    app.state.enable_server_load_tracking = False
    app.state.server_load = 0
    app.state.openai_serving_tokenization = ErrorFactory(code)
    handler = None if mode == "missing" else Handler(mode, code)
    app.state.openai_serving_completion = handler
    app.state.openai_serving_chat = handler
    app.include_router(completion_router.router)
    app.include_router(chat_router.router)
    return app


ENDPOINTS = ["/v1/completions", "/v1/completions/render",
             "/v1/chat/completions", "/v1/chat/completions/render"]


def body(endpoint, stream=False):
    result = dict(model="target", max_tokens=1, temperature=0, stream=stream)
    if "/chat/" in endpoint:
        result["messages"] = [{"role": "user", "content": "hello"}]
    else:
        result["prompt"] = [1, 2]
    return result


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("mode,code", [("raise", 503), ("missing", 400),
                                      ("returned_error", 413)])
def test_error_body_and_http_status_agree(endpoint, mode, code):
    with TestClient(app_for(mode, code)) as client:
        response = client.post(endpoint, json=body(endpoint))
    assert response.status_code == code
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_successful_responses_remain_successful(endpoint):
    with TestClient(app_for("success", 500)) as client:
        response = client.post(endpoint, json=body(endpoint))
    assert response.status_code == 200
    if endpoint.endswith("/render"):
        assert response.json() == [{"prompt_token_ids": [1, 2]}]
    else:
        assert response.json()["id"] == "test-id"
        assert response.json()["usage"]["completion_tokens"] == 1


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
def test_streaming_success_keeps_sse_response(endpoint):
    with TestClient(app_for("stream", 500)) as client:
        response = client.post(endpoint, json=body(endpoint, stream=True))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text
