"""Client protocol construction only; fake SSE is not model validation."""
import asyncio
import json
from pathlib import Path

import benchmark


def test_trace_identity_uses_supported_xargs(monkeypatch, tmp_path):
    posted = []

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body
            self.content = self.chunks()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def chunks(self):
            count = self.body['max_tokens']
            obj = dict(id='test-only', choices=[dict(token_ids=list(range(count)), text='', finish_reason='length')], usage=dict(completion_tokens=count))
            yield ('data: ' + json.dumps(obj) + '\n\n').encode()
            yield b'data: [DONE]\n\n'

    class Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, json):
            posted.append(json)
            return Response(json)

    monkeypatch.setattr(benchmark.aiohttp, 'ClientSession', Session)
    monkeypatch.setattr(benchmark.aiohttp, 'TCPConnector', lambda **kwargs: None)
    monkeypatch.delenv('CDB_TRACE_MISSING_FIRST', raising=False)
    rows = [dict(id='source-A', max_tokens=5, prompt_ids=dict(qwen=[1, 2, 3]))]
    output = tmp_path / 'measurement'
    summary = asyncio.run(benchmark.run('http://no-network', rows, 'qwen', 'closed', 1, 1, 42, output))
    assert summary['completed'] == 1 and summary['failed'] == 0
    body = posted[0]
    assert body['vllm_xargs'] == {'cdb_trace_key': 'measurement:source-A'}
    assert 'extra_body' not in body and 'cdb_draft_k' not in body['vllm_xargs']
    assert body['prompt'] == [1, 2, 3] and body['max_tokens'] == 5
    assert body['temperature'] == 0 and body['seed'] == 42
    identity = json.loads((output / 'identities.jsonl').read_text())
    assert identity['trace_key'] == body['vllm_xargs']['cdb_trace_key']
    assert identity['prompt_token_ids'] == body['prompt']
    assert json.loads((output / 'http_requests.jsonl').read_text()) == body

    monkeypatch.setenv('CDB_TRACE_MISSING_FIRST', '1')
    asyncio.run(benchmark.run('http://no-network', rows, 'qwen', 'closed', 1, 1, 42, tmp_path / 'warmup'))
    assert posted[-1]['vllm_xargs']['cdb_trace_key'] == 'intentional_missing:warmup:source-A'
