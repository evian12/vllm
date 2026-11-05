# SPDX-License-Identifier: Apache-2.0
"""Build an exact-context, hindsight K replay from CDB Target verification logs.

Use a max-K reference run. This is a diagnostic oracle for observed contexts,
not a deployable predictor or an end-to-end throughput upper bound.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

PREFIX = "[CDB_REFILL] "


def extract_records(lines):
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
        drafts = event.get("drafts")
        accepted = event.get("accepted")
        count = event.get("context_tokens")
        fp = event.get("fingerprint")
        if (not isinstance(drafts, list) or not drafts
                or type(accepted) is not int
                or not 0 <= accepted <= len(drafts)
                or type(count) is not int or count < 1
                or not isinstance(fp, str) or len(fp) != 32):
            raise ValueError("Malformed p2_verified acceptance evidence")
        yield dict(
            trace_key=key,
            fingerprint=fp,
            context_tokens=count,
            observed_k=len(drafts),
            accepted=accepted,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-k", type=int, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists; use a new path to preserve prior evidence")
    if args.max_k < 1:
        p.error("max-k must be positive")
    from vllm.v1.spec_decode.cdb_av_acceptance import AcceptanceReplay
    with args.server_log.open(errors="replace") as source:
        records = list(extract_records(source))
    if not records:
        p.error("no keyed p2_verified records found; disable CDB_AV_LOW_LOG in the reference run")
    if any(row["observed_k"] > args.max_k for row in records):
        p.error("reference K exceeds --max-k")
    text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records)
    # Validate before publishing the replay file.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        temporary.write_text(text)
        replay = AcceptanceReplay.from_jsonl(temporary, args.max_k)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    counts = Counter()
    for row in records:
        counts["records"] += 1
        counts["full_accept_censored"] += row["accepted"] == row["observed_k"]
        counts["certainly_wasted_suffix_tokens"] += max(
            0, row["observed_k"] - row["accepted"] - 1)
    print(json.dumps(dict(counts, unique_contexts=len(replay.records),
                          ambiguous_contexts=len(replay.ambiguous),
                          output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
