"""CPU-only hook plumbing; no simulated counts are GPU evidence."""
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def test_refill_hook_counts_real_call_inputs_without_tensor_reads(monkeypatch, tmp_path):
    monkeypatch.setenv('BASELINE_OBSERVE_DIRECTORY', str(tmp_path))
    before = list(sys.meta_path)
    path = Path(__file__).with_name('manual_observer.py')
    spec = importlib.util.spec_from_file_location('test_cdb_shape_observer', path)
    observer = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(observer)
        monkeypatch.setattr(observer, 'start', lambda: None)
        calls = []
        sentinel = object()

        class TensorMetadata:
            def __init__(self, n):
                self.n = n

            def numel(self):
                return self.n

            def cpu(self):
                raise AssertionError('Must not read a tensor')

        class Refill:
            def _forward(self, *args):
                calls.append(args)
                return sentinel

        observer.patch_refill(SimpleNamespace(RefillSession=Refill))
        session = Refill()
        ids = TensorMetadata(10)
        reqs = ['a', 'b', 'c', 'd']
        assert session._forward(ids, None, None, None, reqs, 'mixed_prefill_decode') is sentinel
        assert calls[0][0] is ids and calls[0][4] is reqs
        assert session._forward(TensorMetadata(2), None, None, None, ['e'], 'mixed_prefill') is sentinel
        assert observer.STATS['cdb_draft_dispatches'] == 2
        assert observer.STATS['cdb_draft_request_rows'] == 5
        assert observer.STATS['cdb_draft_token_rows'] == 12
        assert observer.STATS['cdb_phase_mixed_prefill_decode_calls'] == 1
        assert observer.STATS['cdb_phase_mixed_prefill_token_rows'] == 2
    finally:
        sys.meta_path[:] = before


def test_target_row_classification_and_ready_schedule_wait(monkeypatch, tmp_path):
    monkeypatch.setenv('BASELINE_OBSERVE_DIRECTORY', str(tmp_path))
    before = list(sys.meta_path)
    path = Path(__file__).with_name('manual_observer.py')
    spec = importlib.util.spec_from_file_location('test_cdb_target_observer', path)
    observer = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(observer)
        monkeypatch.setattr(observer, 'start', lambda: None)
        class Scheduler(SimpleNamespace):
            def add_request(self, item):
                self.waiting.append(item)

            def schedule(self):
                return self.result

        observer.patch_scheduler(SimpleNamespace(Scheduler=Scheduler))
        assert hasattr(Scheduler.schedule, '__wrapped__')
        scheduler = Scheduler(requests={
            'v': SimpleNamespace(num_output_tokens=8),
            'p': SimpleNamespace(num_output_tokens=0),
            'a': SimpleNamespace(num_output_tokens=4),
            'r': SimpleNamespace(num_output_tokens=4),
        }, waiting=['new'], running=[], _cdb_p2=SimpleNamespace(ready={'v': {'ready_at': time.monotonic()-.01}}))
        mixed = SimpleNamespace(num_scheduled_tokens={'v': 3, 'p': 10, 'a': 1}, scheduled_spec_decode_tokens={'v': [1, 2]})
        scheduler.result = mixed
        assert scheduler.schedule() is mixed
        scheduler.result = SimpleNamespace(num_scheduled_tokens={'a': 1}, scheduled_spec_decode_tokens={})
        scheduler.schedule()
        scheduler.result = SimpleNamespace(num_scheduled_tokens={'r': 5}, scheduled_spec_decode_tokens={})
        scheduler.schedule()
        assert observer.STATS['cdb_target_schedules'] == 3
        assert observer.STATS['cdb_target_verify_rows'] == 1
        assert observer.STATS['cdb_target_prefill_rows'] == 1
        assert observer.STATS['cdb_target_ar_rows'] == 2
        assert observer.STATS['cdb_target_other_rows'] == 1
        assert observer.STATS['cdb_target_token_rows'] == 20
        assert observer.STATS['cdb_target_pure_ar_calls'] == 1
        assert observer.STATS['cdb_target_pure_ar_with_waiting'] == 1
        assert observer.STATS['cdb_ready_wait_count'] == 1
        assert observer.STATS['cdb_ready_wait_sum_ms'] >= 10
    finally:
        sys.meta_path[:] = before
