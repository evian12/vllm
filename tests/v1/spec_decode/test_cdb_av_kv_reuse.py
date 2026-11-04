# SPDX-License-Identifier: Apache-2.0
"""Focused correctness tests for CDB-AV accepted-prefix KV reuse."""

from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint
from vllm.v1.spec_decode.cdb_av_kv_reuse import reuse_span


def make_certificate(old_prefix, drafts, generated):
    return {
        "source_fp": token_fingerprint(old_prefix),
        "drafts": list(drafts),
        "generated": list(generated),
    }


def test_reuse_exact_verified_prefix():
    old_prefix = [1, 2, 3]
    drafts = [10, 11, 12]
    generated = [10, 11, 99]  # Two accepted candidates plus replacement/bonus.
    cached_inputs = old_prefix + drafts[:-1]
    prefix = old_prefix + generated

    assert reuse_span(
        cached_inputs, prefix, old_prefix, drafts,
        make_certificate(old_prefix, drafts, generated),
    ) == (len(old_prefix) + 2, None)


def test_reuse_survives_append_only_target_ar_growth():
    old_prefix = [1, 2, 3]
    drafts = [10, 11, 12]
    generated = [10, 11, 99]
    cached_inputs = old_prefix + drafts[:-1]
    # Target produced two additional ordinary AR tokens after verification.
    prefix = old_prefix + generated + [100, 101]

    assert reuse_span(
        cached_inputs, prefix, old_prefix, drafts,
        make_certificate(old_prefix, drafts, generated),
    ) == (len(old_prefix) + 2, None)


def test_reuse_survives_single_ar_token_growth():
    old_prefix = [1, 2]
    drafts = [7, 8]
    generated = [7, 8, 9]
    cached_inputs = old_prefix + drafts[:-1]
    prefix = old_prefix + generated + [10]

    assert reuse_span(
        cached_inputs, prefix, old_prefix, drafts,
        make_certificate(old_prefix, drafts, generated),
    ) == (len(cached_inputs), None)


def test_reuse_zero_accepted_candidates_keeps_only_old_prefix():
    old_prefix = [1, 2, 3]
    drafts = [10, 11, 12]
    generated = [99]
    cached_inputs = old_prefix + drafts[:-1]
    prefix = old_prefix + generated + [100]

    assert reuse_span(
        cached_inputs, prefix, old_prefix, drafts,
        make_certificate(old_prefix, drafts, generated),
    ) == (len(old_prefix), None)


def test_reuse_full_acceptance_is_bounded_by_materialized_inputs():
    old_prefix = [1, 2, 3]
    drafts = [10, 11, 12]
    generated = drafts + [99]
    cached_inputs = old_prefix + drafts[:-1]
    prefix = old_prefix + generated

    # Draft has materialized K-1 candidate inputs; the last candidate and
    # Target bonus token are not part of this cached input lineage.
    assert reuse_span(
        cached_inputs, prefix, old_prefix, drafts,
        make_certificate(old_prefix, drafts, generated),
    ) == (len(old_prefix) + len(drafts) - 1, None)


def test_divergence_inside_certified_commit_falls_back():
    old_prefix = [1, 2, 3]
    drafts = [10, 11, 12]
    generated = [10, 11, 99]
    cached_inputs = old_prefix + drafts[:-1]
    prefix = old_prefix + [10, 77, 99, 100]

    assert reuse_span(
        cached_inputs, prefix, old_prefix, drafts,
        make_certificate(old_prefix, drafts, generated),
    ) == (0, "commit_prefix")


def test_cached_input_lineage_mismatch_falls_back():
    old_prefix = [1, 2, 3]
    drafts = [10, 11, 12]
    generated = [10, 11, 99]
    cached_inputs = old_prefix + [10, 77]
    prefix = old_prefix + generated

    assert reuse_span(
        cached_inputs, prefix, old_prefix, drafts,
        make_certificate(old_prefix, drafts, generated),
    ) == (0, "cached_input_lineage")


def test_stale_source_fingerprint_falls_back():
    old_prefix = [1, 2, 3]
    drafts = [10, 11, 12]
    generated = [10, 11, 99]
    certificate = make_certificate([1, 2, 4], drafts, generated)

    assert reuse_span(
        old_prefix + drafts[:-1], old_prefix + generated,
        old_prefix, drafts, certificate,
    ) == (0, "provenance")


def test_empty_generated_sequence_falls_back():
    old_prefix = [1, 2, 3]
    drafts = [10, 11, 12]
    certificate = make_certificate(old_prefix, drafts, [])

    assert reuse_span(
        old_prefix + drafts[:-1], old_prefix,
        old_prefix, drafts, certificate,
    ) == (0, "commit_prefix")
