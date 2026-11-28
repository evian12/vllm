# SPDX-License-Identifier: Apache-2.0
"""Exercise error HTTP contracts through the actual auxiliary API routers."""
import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.engine.protocol import ErrorResponse


class Handler:
    def __init__(self, mode, code):
        self.mode, self.code = mode, code

    def create_error_response(self, message):
        return ErrorResponse(
            error=dict(
                message=str(message), type="TestError", param=None, code=self.code
            )
        )

    def __getattr__(self, name):
        async def reply(*args, **kwargs):
            if self.mode == "raise":
                raise RuntimeError("simulated auxiliary handler failure")
            return self.create_error_response("simulated returned error")

        return reply


# module, app-state handler, HTTP method, URL, valid request body
CASES = [
    (
        "openai.responses",
        "openai_serving_responses",
        "POST",
        "/v1/responses",
        {"model": "target", "input": "hello"},
    ),
    ("openai.responses", "openai_serving_responses", "GET", "/v1/responses/test", None),
    (
        "openai.responses",
        "openai_serving_responses",
        "POST",
        "/v1/responses/test/cancel",
        None,
    ),
    (
        "openai.translations",
        "openai_serving_transcription",
        "POST",
        "/v1/audio/transcriptions",
        "audio",
    ),
    (
        "openai.translations",
        "openai_serving_translation",
        "POST",
        "/v1/audio/translations",
        "audio",
    ),
    (
        "pooling.score",
        "openai_serving_scores",
        "POST",
        "/score",
        {"model": "target", "text_1": "hello", "text_2": "world"},
    ),
    (
        "pooling.score",
        "openai_serving_scores",
        "POST",
        "/rerank",
        {"model": "target", "query": "hello", "documents": ["world"]},
    ),
    (
        "pooling.pooling",
        "openai_serving_pooling",
        "POST",
        "/pooling",
        {"model": "target", "input": "hello"},
    ),
    (
        "pooling.classify",
        "openai_serving_classification",
        "POST",
        "/classify",
        {"model": "target", "input": "hello"},
    ),
    (
        "pooling.embed",
        "openai_serving_embedding",
        "POST",
        "/v1/embeddings",
        {"model": "target", "input": "hello"},
    ),
    (
        "serve.disagg",
        "serving_tokens",
        "POST",
        "/inference/v1/generate",
        {
            "model": "target",
            "token_ids": [1, 2],
            "sampling_params": {"temperature": 0, "max_tokens": 1},
        },
    ),
    (
        "serve.tokenize",
        "openai_serving_tokenization",
        "POST",
        "/tokenize",
        {"model": "target", "prompt": "hello"},
    ),
]


@pytest.mark.parametrize(
    "module,state,method,url,body", CASES, ids=[c[3] for c in CASES]
)
@pytest.mark.parametrize(
    "mode,code", [("raise", 503), ("missing", 400), ("returned_error", 413)]
)
def test_auxiliary_error_http_status(module, state, method, url, body, mode, code):
    if state == "openai_serving_tokenization" and mode == "missing":
        pytest.skip(
            "Tokenization is a required base handler, not an optional model API"
        )
    router = importlib.import_module(f"vllm.entrypoints.{module}.api_router")
    app = FastAPI()
    app.state.enable_server_load_tracking = False
    app.state.server_load = 0
    app.state.openai_serving_tokenization = Handler(mode, code)
    setattr(app.state, state, None if mode == "missing" else Handler(mode, code))
    app.include_router(router.router)
    with TestClient(app) as client:
        if body == "audio":
            response = client.request(
                method,
                url,
                data={"model": "target"},
                files={"file": ("test.wav", b"audio", "audio/wav")},
            )
        else:
            response = client.request(method, url, json=body)
    assert response.status_code == code, response.text
    assert response.json()["error"]["code"] == code
