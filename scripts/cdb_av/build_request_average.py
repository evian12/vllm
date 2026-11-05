# SPDX-License-Identifier: Apache-2.0
"""Build one historical Target acceptance mean per logical request.

This is an experimental hindsight policy. The reference run and evaluated run
must use the same request set. An unobserved request falls back to online HF.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint
from vllm.v1.spec_decode.cdb_av_request_average import RequestAverageK, mean_to_k

PREFIX = "[CDB_REFILL] "


def client_prompts(lines):
    prompts = {}
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key, tokens = row.get("trace_key"), row.get("prompt_token_ids")
        if not isinstance(key, str) or not key or not isinstance(tokens, list):
            continue
        fp = token_fingerprint(tokens)
        if key in prompts and prompts[key] != fp:
            raise ValueError(f"Conflicting client prompt for {key}")
        prompts[key] = fp
    if not prompts:
        raise ValueError("Client log has no keyed prompt_token_ids")
    return prompts


def accepted_by_request(lines, max_k):
    result = defaultdict(lambda: dict(accepted_total=0, verified_count=0,
                                      request_ids=set(), full_accept_count=0))
    for line in lines:
        if PREFIX not in line:
            continue
        try:
            event = json.loads(line.split(PREFIX, 1)[1])
        except json.JSONDecodeError:
            continue
        if event.get("event") != "p2_verified":
            continue
        key = event.get("trace_key")
        if not isinstance(key, str) or not key:
            continue
        accepted, drafts, req_id = (event.get("accepted"), event.get("drafts"),
                                    event.get("req_id"))
        if (type(accepted) is not int or not isinstance(drafts, list)
                or not 0 <= accepted <= len(drafts) <= max_k
                or not isinstance(req_id, str) or not req_id):
            raise ValueError(f"Malformed Target verification for {key}")
        item = result[key]
        item["accepted_total"] += accepted
        item["verified_count"] += 1
        item["full_accept_count"] += accepted == len(drafts)
        item["request_ids"].add(req_id)
    for key, item in result.items():
        if len(item["request_ids"]) != 1:
            raise ValueError(f"Multiple physical request IDs for {key}")
    if not result:
        raise ValueError("Server log has no keyed p2_verified observations")
    return result


def build_rows(server_lines, client_lines, max_k):
    prompts = client_prompts(client_lines)
    records = accepted_by_request(server_lines, max_k)
    if missing := records.keys() - prompts.keys():
        raise ValueError(f"Verified requests absent from client prompts: {sorted(missing)}")
    rows = []
    for key, item in sorted(records.items()):
        total, count = item["accepted_total"], item["verified_count"]
        rows.append(dict(trace_key=key, prompt_fingerprint=prompts[key],
                         accepted_total=total, verified_count=count,
                         mean_accepted=total / count,
                         chosen_k=mean_to_k(total, count, max_k),
                         server_max_k=max_k,
                         full_accept_count=item["full_accept_count"]))
    return rows, len(prompts) - len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--client-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-k", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    with args.server_log.open(errors="replace") as source, args.client_log.open() as client:
        rows, no_verify = build_rows(source, client, args.max_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        temporary.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                                     for row in rows))
        RequestAverageK.from_jsonl(temporary, args.max_k)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(dict(output=str(args.output), observed_requests=len(rows),
                          requests_without_verification=no_verify,
                          k_distribution={str(k): sum(r["chosen_k"] == k for r in rows)
                                          for k in sorted(set(r["chosen_k"] for r in rows))}),
                     indent=2))


if __name__ == "__main__":
    main()
