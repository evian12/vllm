#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small HTTP smoke to exercise CDB Draft eager/replay; not a TPS benchmark."""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:18084/v1/chat/completions"
N = 32
CONCURRENCY = 4
PROMPT = (
    "Write a detailed explanation of speculative decoding for LLM serving. "
    "Explain draft generation, target verification, accepted prefixes, "
    "rejected tokens, and how speculative decoding improves throughput."
)


def send_request(index):
    payload = {
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": 128,
    }
    request = urllib.request.Request(
        URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            reply = json.loads(response.read())
        completion = reply.get("choices", [])
        if not completion:
            raise RuntimeError("Missing response choices")
        return index, True, reply.get("usage", {}).get("completion_tokens", 0), ""
    except Exception as exc:
        return index, False, 0, str(exc)


def main():
    print(f"CDB Draft profile smoke: requests={N} concurrency={CONCURRENCY}", flush=True)
    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(send_request, i) for i in range(N)]
        for completed in as_completed(futures):
            result = completed.result()
            results.append(result)
            print(f"request={result[0]} ok={result[1]} tokens={result[2]} {result[3]}",
                  flush=True)
    ok = sum(item[1] for item in results)
    print(f"Completed: {ok}/{N}; generated tokens: {sum(x[2] for x in results)}")
    print(f"Diagnostic wall time (not production TPS): {time.perf_counter()-start:.3f}s")
    return 0 if ok == N else 1


if __name__ == "__main__":
    raise SystemExit(main())
