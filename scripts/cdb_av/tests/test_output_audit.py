"""Output audit must work with the production observer's plain-dict protocol."""
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest

spec = importlib.util.spec_from_file_location(
    'output_audit', Path(__file__).resolve().parents[1] / 'cdb_ar_output_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class OutputAuditTests(unittest.TestCase):
    def fixture(self, strict=False):
        observer = NS(STATS={}, LOCK=threading.Lock(), start=lambda: None)
        # This deliberately matches the production observer: unseen keys fail.
        def record(key, value=1):
            observer.STATS[key] += value
        observer.record = record
        class Scheduler:
            def update_from_output(self, scheduled, output):
                return output
        observer.PATCHES = {'vllm.v1.core.sched.scheduler': lambda module: None}
        audit.attach(observer, strict=strict)
        observer.PATCHES['vllm.v1.core.sched.scheduler'](NS(Scheduler=Scheduler))
        scheduler = Scheduler()
        scheduler._cdb_p2 = NS(_draft_wait_fallbacks={'r': ('kv_resource',)})
        scheduler.requests = {'r': NS(num_output_tokens=0)}
        scheduled = NS(num_scheduled_tokens={'r': 1}, scheduled_spec_decode_tokens={})
        return observer, scheduler, scheduled

    @staticmethod
    def output(tokens):
        return {0: NS(outputs=[NS(request_id='r', new_token_ids=tokens)])}

    def test_first_verify_empty_and_dynamic_reason_registration(self):
        observer, scheduler, scheduled = self.fixture()
        scheduler.update_from_output(scheduled, self.output([7]))
        self.assertEqual(observer.STATS, {'cdb_output_tokens_audited': 1,
                                         'cdb_output_first_tokens': 1})
        scheduler.requests['r'].num_output_tokens = 1
        scheduled.scheduled_spec_decode_tokens = {'r': [8, 9]}
        scheduler.update_from_output(scheduled, self.output([8, 9]))
        scheduled.scheduled_spec_decode_tokens = {}
        scheduler.update_from_output(scheduled, self.output([]))
        scheduler.update_from_output(scheduled, self.output([11]))
        self.assertEqual(observer.STATS, {'cdb_output_tokens_audited': 4,
            'cdb_output_first_tokens': 1, 'cdb_output_verify_tokens': 2,
            'cdb_output_ar_fallback_tokens': 1,
            'cdb_output_ar_fallback_kv_resource_tokens': 1})

    def test_strict_fallback_is_counted_then_rejected(self):
        observer, scheduler, scheduled = self.fixture(strict=True)
        scheduler.requests['r'].num_output_tokens = 1
        with self.assertRaisesRegex(RuntimeError, 'AR fallback tokens'):
            scheduler.update_from_output(scheduled, self.output([11]))
        self.assertEqual(observer.STATS['cdb_output_ar_fallback_kv_resource_tokens'], 1)

    def test_non_cdb_scheduler_passes_through_without_counters(self):
        observer, scheduler, scheduled = self.fixture(strict=True)
        del scheduler._cdb_p2
        output = self.output([7, 8])
        self.assertIs(scheduler.update_from_output(scheduled, output), output)
        self.assertEqual(observer.STATS, {})


if __name__ == '__main__':
    unittest.main()
