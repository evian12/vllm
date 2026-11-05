# SPDX-License-Identifier: Apache-2.0
"""Build an exact-context, hindsight K replay from CDB Target verification logs.

Use a max-K reference run. This is a diagnostic oracle for observed contexts,
not a deployable predictor or an end-to-end throughput upper bound.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint

PREFIX = "[CDB_REFILL] "


def extract_records(lines, include_candidates=False):
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
        row = dict(
            trace_key=key,
            fingerprint=fp,
            context_tokens=count,
            observed_k=len(drafts),
            accepted=accepted,
        )
        if include_candidates:
            generated = event.get("generated")
            if (not isinstance(generated, list)
                    or len(generated) < accepted
                    or any(type(token) is not int or token < 0
                           for token in drafts + generated)):
                raise ValueError("Malformed Target generated-token evidence")
            row["drafts"] = drafts
            row["generated"] = generated
        yield row


def load_client_tokens(lines):
    """Read exact token IDs returned by vLLM, keyed by the stable trace key."""
    result = {}
    for line in lines:
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        key = row.get("trace_key")
        prompt = row.get("prompt_token_ids")
        output = row.get("generated_token_ids")
        if (not row.get("ok") or not isinstance(key, str) or not key
                or not isinstance(prompt, list) or not prompt
                or not isinstance(output, list)
                or any(type(token) is not int or token < 0
                       for token in prompt + output)):
            raise ValueError("Missing exact client token-ID evidence")
        if key in result:
            raise ValueError(f"Duplicate client trace key: {key}")
        result[key] = prompt + output
    if not result:
        raise ValueError("No client token-ID evidence")
    return result


def expand_accepted_prefixes(records, client_tokens):
    """Shift only through Target-accepted candidates of a real verification.

    At shift j, the original Draft's remaining candidate chain and Target's
    accepted chain are valid conditional continuations of the same prefix.
    The first rejection remains the first rejection; full-accept results stay
    right-censored. Exact token IDs and fingerprint are checked before use.
    """
    for row in records:
        key = row["trace_key"]
        full = client_tokens.get(key)
        if full is None:
            raise ValueError(f"No client tokens for verified trace key {key}")
        start = row["context_tokens"]
        drafts = row["drafts"]
        accepted = row["accepted"]
        generated = row["generated"]
        if (start > len(full)
                or token_fingerprint(full[:start]) != row["fingerprint"]
                or generated[:accepted] != drafts[:accepted]):
            raise ValueError(f"Target/Client accepted-prefix mismatch: {key}")
        # max_tokens may truncate the API result inside an all-accepted
        # verification. Only prefixes actually present in the response are
        # eligible as future formal contexts.
        available = min(accepted, len(full) - start)
        if full[start:start + available] != drafts[:available]:
            raise ValueError(f"Target/Client accepted-prefix mismatch: {key}")
        for shift in range(available + 1):
            remaining = drafts[shift:]
            if not remaining:
                continue  # An all-accepted suffix has no observed candidate.
            prefix = full[:start + shift]
            yield dict(trace_key=key,
                       fingerprint=token_fingerprint(prefix),
                       context_tokens=start + shift,
                       observed_k=len(remaining),
                       accepted=accepted - shift,
                       drafts=remaining,
                       derived_shift=shift)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--client-log", type=Path,
                   help="Expand accepted intermediate prefixes using exact token IDs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-k", type=int, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists; use a new path to preserve prior evidence")
    if args.max_k < 1:
        p.error("max-k must be positive")
    from vllm.v1.spec_decode.cdb_av_acceptance import AcceptanceReplay
    with args.server_log.open(errors="replace") as source:
        records = list(extract_records(
            source, include_candidates=args.client_log is not None))
    if not records:
        p.error("no keyed p2_verified records found; disable CDB_AV_LOW_LOG in the reference run")
    reference_verifications = len(records)
    reference_wasted_suffix = sum(max(
        0, row["observed_k"] - row["accepted"] - 1) for row in records)
    if args.client_log is not None:
        with args.client_log.open() as client:
            client_tokens = load_client_tokens(client)
        records = list(expand_accepted_prefixes(records, client_tokens))
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
    print(json.dumps(dict(counts, reference_verifications=reference_verifications,
                          reference_wasted_suffix=reference_wasted_suffix,
                          derived_intermediate_records=len(records) - reference_verifications,
                          unique_contexts=len(replay.records),
                          ambiguous_contexts=len(replay.ambiguous),
                          output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
