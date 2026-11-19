# SPDX-License-Identifier: Apache-2.0
"""Historical-policy fallbacks must be explicit without changing online HF."""
from types import SimpleNamespace

import pytest

from vllm.v1.spec_decode.cdb_av_acceptance import (
    AcceptanceReplay, should_stop,
)
from vllm.v1.spec_decode.cdb_av_config import get_history_k_fallback
from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint
from vllm.v1.spec_decode.cdb_av_persistent import PersistentDraftQueue
from vllm.v1.spec_decode.cdb_av_request_average import RequestAverageK


def test_fallback_default_and_options(monkeypatch):
    monkeypatch.delenv('CDB_AV_HISTORY_K_FALLBACK', raising=False)
    assert get_history_k_fallback() == 'hf_confidence'
    for mode in ['hf_confidence', 'fixed_max', ' fixed_max ']:
        monkeypatch.setenv('CDB_AV_HISTORY_K_FALLBACK', mode)
        assert get_history_k_fallback() == mode.strip()


@pytest.mark.parametrize('mode', ['', '0', '1', 'fixed', 'HF_CONFIDENCE', 'nan'])
def test_fallback_rejects_invalid_mode(monkeypatch, mode):
    monkeypatch.setenv('CDB_AV_HISTORY_K_FALLBACK', mode)
    with pytest.raises(ValueError, match='CDB_AV_HISTORY_K_FALLBACK'):
        get_history_k_fallback()


def queue_and_request(monkeypatch, fallback='fixed_max'):
    monkeypatch.setenv('CDB_AV_OUTPUT_BUDGET_CAP', '0')
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.max_k = 8
    queue.acceptance_replay = None
    queue.history_k_fallback = fallback
    fp = token_fingerprint([1, 2, 3])
    queue.request_average = RequestAverageK({'known': (fp, 3, 2.5, 4)}, 8)
    request = SimpleNamespace(request_id='r', prompt_token_ids=[1, 2, 3],
                              sampling_params=SimpleNamespace(
                                  extra_args={'cdb_trace_key': 'known'}))
    return queue, request


@pytest.mark.parametrize('fallback', ['fixed_max', 'hf_confidence'])
def test_matched_mean_is_unchanged(monkeypatch, fallback):
    queue, request = queue_and_request(monkeypatch, fallback)
    assert queue.resolve_k_cap(request, None) == (3, 'request_average')


@pytest.mark.parametrize('case', ['no_key', 'missing_key', 'prompt_mismatch'])
@pytest.mark.parametrize('fallback,source', [
    ('fixed_max', 'fixed_max_fallback'), ('hf_confidence', 'hf_fallback'),
])
def test_average_misses_use_selected_fallback(monkeypatch, case, fallback, source):
    queue, request = queue_and_request(monkeypatch, fallback)
    if case == 'no_key':
        request.sampling_params.extra_args = {}
    elif case == 'missing_key':
        request.sampling_params.extra_args['cdb_trace_key'] = 'unknown'
    else:
        request.prompt_token_ids = [4, 5, 6]
    assert queue.resolve_k_cap(request, None) == (8, source)
    assert should_stop('request_avg_acceptance', source, True) == (fallback == 'hf_confidence')


def test_old_queue_fixture_retains_hf_fallback(monkeypatch):
    queue, request = queue_and_request(monkeypatch)
    del queue.history_k_fallback
    request.sampling_params.extra_args = {}
    assert queue.resolve_k_cap(request, None) == (8, 'hf_fallback')


def test_exact_replay_miss_uses_fixed_fallback(monkeypatch):
    queue, request = queue_and_request(monkeypatch)
    queue.request_average = None
    queue.acceptance_replay = AcceptanceReplay({}, set())
    result = queue.resolve_k_cap(request, {'fingerprint': 'a' * 32, 'context_tokens': 4})
    assert result == (8, 'fixed_max_fallback')
    assert not should_stop('replay_acceptance', result[1], True)


def test_fixed_history_fallback_does_not_disable_online_hf(monkeypatch):
    queue, request = queue_and_request(monkeypatch)
    queue.request_average = None
    monkeypatch.setenv('CDB_AV_K_POLICY', 'hf_confidence')
    assert queue.resolve_k_cap(request, None) == (8, 'configured')
    assert should_stop('hf_confidence', 'configured', True)
    assert should_stop('hf_confidence', 'fixed_max_fallback', True)


@pytest.mark.parametrize('cap', [1, 8])
def test_historical_boundary_caps_are_preserved(monkeypatch, cap):
    queue, request = queue_and_request(monkeypatch)
    fp = token_fingerprint(request.prompt_token_ids)
    queue.request_average = RequestAverageK({'known': (fp, cap, float(cap), 1)}, 8)
    assert queue.resolve_k_cap(request, None) == (cap, 'request_average')


def test_low_log_reports_worker_ready_source(monkeypatch):
    import json
    from collections import Counter
    from vllm.v1.spec_decode import cdb_av_refill as refill

    monkeypatch.setenv('CDB_AV_LOW_LOG', '1')
    monkeypatch.setenv('CDB_AV_REFILL_TRACE', '0')
    for name in ['_LOW_LOG_K', '_LOW_LOG_K_SOURCES', '_LOW_LOG_REASONS',
                 '_LOW_LOG_REUSE', '_LOW_LOG_REPLAY']:
        monkeypatch.setattr(refill, name, Counter())
    monkeypatch.setattr(refill, '_LOW_LOG_COUNTS', Counter(__total__=1023))
    logged = []
    monkeypatch.setattr(refill.logger, 'info', lambda fmt, value: logged.append(value))
    refill.emit('ready', k_actual=3, reason='max_k', k_source='request_average')
    assert json.loads(logged[0])['k_sources'] == {'request_average': 1}
