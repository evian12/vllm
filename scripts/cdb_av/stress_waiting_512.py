# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import hashlib
import json
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor


PROMPTS = [
    "Explain transformer attention with an example.",
    "Explain KV cache in large language model inference.",
    "Explain continuous batching and its scheduling policies.",
    "Explain speculative decoding step by step.",
    "Explain the difference between MHA, GQA and MQA.",
    "Explain how CUDA Graph improves LLM inference.",
    "Explain memory bandwidth bottlenecks in autoregressive decoding.",
    "Explain how FlashAttention reduces GPU memory traffic.",
    "Explain request preemption in vLLM.",
    "Explain chunked prefill and continuous batching.",
    "Explain prefix caching in LLM serving.",
    "Explain dynamic speculative decoding lengths.",
    "Explain the difference between prefill and decode phases.",
    "Explain how GPU KV block allocation works.",
    "Explain why heterogeneous draft lengths cause scheduling challenges.",
    "Explain the design of an asynchronous verification scheduler.",
]


def build_payload(index, max_tokens, request_id=None, draft_k=8,
                  capture_token_ids=False):
    """Build raw HTTP JSON; SDK-only extra_body is not a wire field."""
    prompt = PROMPTS[index % len(PROMPTS)]
    prompt = prompt + "\n" + (
        "Discuss the design, limitations, implementation and examples. "
        * (1 + index % 4)
    )

    payload = {
        "model": "qwen3-8b",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "vllm_xargs": {
            "cdb_draft_k": draft_k,
            # Stable across repeated runs; the context fingerprint still has
            # to match before any historical acceptance can be replayed.
            "cdb_trace_key": f"stress512:{index}",
        },
    }

    if request_id is not None:
        payload["request_id"] = request_id
    if capture_token_ids:
        payload["return_token_ids"] = True
    return payload


def send_request(index, port, max_tokens, run_id=None, dry_run=False,
                 draft_k=8, capture_token_ids=False):
    request_id = f"p0_{run_id or uuid.uuid4().hex}_{index}"
    payload = build_payload(index, max_tokens, request_id, draft_k,
                            capture_token_ids)
    evidence = {
        "client_request_id": request_id,
        "trace_key": payload["vllm_xargs"]["cdb_trace_key"],
        "requested_K": payload["vllm_xargs"]["cdb_draft_k"],
        # The completion API has no Worker Draft telemetry. Never infer these
        # from requested_K, output length, or Target finish_reason.
        "effective_K": None,
        "actual_draft_steps": None,
        "stop_reason": None,
        "draft_evidence_status": "unavailable_in_completion_response",
    }
    if dry_run:
        return dict(index=index, k=evidence["requested_K"], dry_run=True,
                    payload=payload, **evidence)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    started = time.monotonic()

    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            result = json.load(response)

        choices = result.get("choices") if isinstance(result, dict) else None
        usage = result.get("usage") if isinstance(result, dict) else None
        if (not isinstance(choices, list) or not choices
                or not isinstance(choices[0], dict)
                or not isinstance(choices[0].get("text"), str)
                or not isinstance(result.get("id"), str)
                or not isinstance(usage, dict)
                or not isinstance(usage.get("completion_tokens"), int)):
            raise ValueError(f"Invalid completion response: {str(result)[:500]}")
        token_evidence = {}
        if capture_token_ids:
            prompt_ids = choices[0].get("prompt_token_ids")
            output_ids = choices[0].get("token_ids")
            if (not isinstance(prompt_ids, list) or not prompt_ids
                    or not isinstance(output_ids, list)
                    or len(output_ids) != usage["completion_tokens"]
                    or any(type(token) is not int or token < 0
                           for token in prompt_ids + output_ids)):
                raise ValueError("Invalid exact token-ID evidence in completion")
            token_evidence = dict(prompt_token_ids=prompt_ids,
                                  generated_token_ids=output_ids)

        elapsed = time.monotonic() - started

        return {
            **evidence,
            "index": index,
            "k": evidence["requested_K"],
            "ok": True,
            "elapsed": elapsed,
            "started_at": started,
            "completed_at": time.monotonic(),
            "output_tokens": usage.get("completion_tokens", 0),
            "output_sha256": hashlib.sha256(
                choices[0]["text"].encode("utf-8")
            ).hexdigest(),
            "response_id": result.get("id"),
            "generation_finish_reason": choices[0].get("finish_reason"),
            **token_evidence,
        }

    except Exception as exc:
        return {
            **evidence,
            "index": index,
            "k": evidence["requested_K"],
            "ok": False,
            "error": str(exc),
            "elapsed": time.monotonic() - started,
            "started_at": started,
            "completed_at": time.monotonic(),
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--run-id", default=None,
                        help="Stable request-ID prefix for paired experiments")
    parser.add_argument("--draft-k", type=int, default=8,
                        help="Client cap for manual-K reference collection")
    parser.add_argument("--capture-token-ids", action="store_true",
                        help="Retain exact prompt/output token IDs for trace expansion")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print request payloads without contacting any server")
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens) < 1:
        parser.error("requests, concurrency and max-tokens must be positive")
    if not 1 <= args.draft_k <= 8:
        parser.error("draft-k must be in [1, 8]")
    run_id = args.run_id or uuid.uuid4().hex
    if args.dry_run:
        for i in range(args.requests):
            print(json.dumps(send_request(i, args.port, args.max_tokens,
                                          run_id, True, args.draft_k,
                                          args.capture_token_ids), ensure_ascii=False))
        return

    started = time.monotonic()
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            loop.run_in_executor(
                pool, send_request,
                i, args.port, args.max_tokens, run_id, False,
                args.draft_k, args.capture_token_ids
            )
            for i in range(args.requests)
        ]

        results = await asyncio.gather(*futures)

    for result in results:
        print(json.dumps(result, ensure_ascii=False))

    total = time.monotonic() - started
    completed = sum(x["ok"] for x in results)
    tokens = sum(x.get("output_tokens", 0) for x in results)

    print("\n===== SUMMARY =====")
    print(f"completed={completed}/{args.requests}")
    print(f"wall_time={total:.3f}s")
    print(f"output_tokens={tokens}")
    print(f"aggregate_tps={tokens / total:.3f}")
    if completed != args.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
