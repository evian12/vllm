# SPDX-License-Identifier: Apache-2.0
"""Full typed token comparisons for already-validated waiting prefixes.

No identity/length-only hits. Snapshots are bounded immutable tuples, contain
no request/KV references, and are only created AFTER the original hasher passed.
"""
from vllm.v1.spec_decode.cdb_av_context_cache import NATIVE_TYPES, _NATIVE_TYPES


def exact_ints(tokens):
    if NATIVE_TYPES:
        return _NATIVE_TYPES.all_builtin_ints(tokens)
    return all(type(token) is int for token in tokens)


def snapshot(req):
    prompt, output = tuple(req.prompt_token_ids), tuple(req.output_token_ids)
    if len(prompt) + len(output) > 1024:
        return None
    if not exact_ints(prompt) or not exact_ints(output):
        return None
    return prompt, output


def matches(req, saved):
    """Compare every token and its exact builtin type, including same-size edits."""
    prompt, output = tuple(req.prompt_token_ids), tuple(req.output_token_ids)
    return (exact_ints(prompt) and exact_ints(output)
            and prompt == saved[0] and output == saved[1])
