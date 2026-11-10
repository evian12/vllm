from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cdb_av.scheduler_sim.phase_aware import (
    FixedHPhasePolicy,
    JointPhasePolicy,
    LengthAwareTargetBatcher,
    PhaseAction,
    PhaseAwareSimulator,
    PhaseConfig,
    PhaseExecutionProfile,
    PhaseRequest,
    QueuePressurePhasePolicy,
    RequestStatus,
    SarathiTargetBatcher,
    VllmChunkedTargetBatcher,
    WorkKind,
    generate_phase_workload,
)


def make_request(
    request_id: str,
    *,
    arrival: float = 0.0,
    prompt: int = 128,
    output: int = 4,
    speculative: bool = True,
    cap: int = 2,
    acceptance: tuple[float, ...] = (1.0,) * 8,
    confidence: tuple[float, ...] = (1.0,) * 8,
) -> PhaseRequest:
    return PhaseRequest(
        request_id=request_id,
        arrival_time_ms=arrival,
        prompt_tokens=prompt,
        max_output_tokens=output,
        draft_cap=cap,
        acceptance_probs=acceptance,
        confidence_by_position=confidence,
        speculative=speculative,
    )


class PhaseAwareSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = PhaseExecutionProfile(
            draft_base_ms=1.0,
            draft_per_request_ms=0.1,
            draft_per_context_token_ms=0.0,
            target_base_ms=1.0,
            target_per_request_ms=0.1,
            prefill_per_token_ms=0.001,
            prefill_context_per_token_ms=0.0,
            verify_per_candidate_ms=0.05,
            verify_context_per_token_ms=0.0,
            ar_per_request_ms=0.1,
            ar_context_per_token_ms=0.0,
        )

    def test_prefill_is_chunked_and_does_not_emit_first_token_early(self):
        req = make_request("long", prompt=1024, output=3, cap=1)
        config = PhaseConfig(
            max_num_seqs=4,
            max_num_batched_tokens=512,
            confidence_threshold=0.0,
        )
        sim = PhaseAwareSimulator(
            [req], config, self.profile, FixedHPhasePolicy(1)
        )

        first = sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(first.prefill_requests, 1)
        self.assertEqual(req.prefill_progress, 512)
        self.assertEqual(req.committed_tokens, 0)
        self.assertEqual(req.status, RequestStatus.PREFILL)

        sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(req.prefill_progress, 1024)
        self.assertEqual(req.committed_tokens, 1)
        self.assertIsNotNone(req.first_token_time_ms)
        self.assertEqual(req.status, RequestStatus.ACTIVE_DRAFT)

    def test_speculative_request_with_one_token_left_falls_back_to_ar(self):
        req = make_request("last-token", prompt=128, output=2, cap=4)
        sim = PhaseAwareSimulator(
            [req],
            PhaseConfig(confidence_threshold=0.0),
            self.profile,
            QueuePressurePhasePolicy(),
        )
        sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(req.committed_tokens, 1)
        self.assertEqual(req.remaining_output_tokens, 1)
        self.assertEqual(req.status, RequestStatus.ACTIVE_AR)
        self.assertEqual(len(sim.active_draft), 0)
        sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(req.committed_tokens, 2)
        self.assertEqual(req.status, RequestStatus.FINISHED)

    def test_one_target_forward_mixes_prefill_verify_and_ar(self):
        requests = [
            make_request("spec-a", output=3, cap=1),
            make_request("ar-b", output=3, speculative=False),
            make_request("spec-c", output=3, cap=1),
            make_request("new-d", arrival=0.1, output=3, cap=1),
        ]
        config = PhaseConfig(
            max_num_seqs=4,
            max_num_batched_tokens=2048,
            confidence_threshold=0.0,
        )
        sim = PhaseAwareSimulator(
            requests, config, self.profile, FixedHPhasePolicy(1)
        )

        initial = sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(initial.prefill_requests, 3)
        self.assertEqual(len(sim.active_draft), 2)
        self.assertEqual(len(sim.ar_queue), 1)
        self.assertEqual(len(sim.prefill_queue), 1)

        sim.execute_action(PhaseAction.DRAFT)
        self.assertEqual(len(sim.ready_queue), 2)

        plan = sim.policy.batcher.plan(sim)
        self.assertEqual(
            {item.kind for item in plan},
            {WorkKind.PREFILL, WorkKind.VERIFY, WorkKind.AR},
        )
        record = sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(record.prefill_requests, 1)
        self.assertEqual(record.verify_requests, 2)
        self.assertEqual(record.ar_requests, 1)
        self.assertEqual(record.request_count, 4)
        self.assertLessEqual(record.target_tokens, config.max_num_batched_tokens)

    def test_target_batch_respects_request_and_token_budgets(self):
        requests = [
            make_request(f"p-{i}", prompt=512, output=2, speculative=False)
            for i in range(8)
        ]
        config = PhaseConfig(max_num_seqs=3, max_num_batched_tokens=700)
        sim = PhaseAwareSimulator(
            requests,
            config,
            self.profile,
            QueuePressurePhasePolicy(SarathiTargetBatcher(chunk_tokens=256)),
        )
        plan = sim.policy.batcher.plan(sim)
        self.assertLessEqual(len(plan), 3)
        self.assertLessEqual(sum(item.token_count for item in plan), 700)
        self.assertTrue(all(item.kind is WorkKind.PREFILL for item in plan))
        self.assertTrue(all(item.token_count <= 256 for item in plan))

    def test_prefill_wait_age_resets_after_each_chunk(self):
        req = make_request("long-wait", prompt=1024, output=2, speculative=False)
        batcher = LengthAwareTargetBatcher(
            min_prefill_budget=256,
            max_prefill_budget=1024,
            long_chunk_tokens=256,
        )
        sim = PhaseAwareSimulator(
            [req],
            PhaseConfig(prefill_wait_budget_ms=12.0),
            self.profile,
            QueuePressurePhasePolicy(batcher),
        )
        sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(req.prefill_progress, 256)
        self.assertAlmostEqual(sim.oldest_prefill_wait_ms(), 0.0)

    def test_verify_candidate_batch_is_atomic(self):
        req = make_request("verify-atomic", output=8, cap=3)
        sim = PhaseAwareSimulator(
            [req], PhaseConfig(), self.profile, QueuePressurePhasePolicy()
        )
        sim.execute_action(PhaseAction.TARGET)
        req.status = RequestStatus.READY
        req.draft_depth = 3
        req.ready_since_ms = sim.now_ms
        sim.active_draft.clear()
        sim.ready_queue.append(req)
        plan = sim.policy.batcher.plan(sim)
        verify_items = [item for item in plan if item.kind is WorkKind.VERIFY]
        self.assertEqual(len(verify_items), 1)
        self.assertEqual(verify_items[0].token_count, 4)

    def test_length_aware_batcher_groups_similar_prompt_lengths(self):
        requests = [
            make_request("a", prompt=128, output=2, speculative=False),
            make_request("b", prompt=256, output=2, speculative=False),
            make_request("c", prompt=300, output=2, speculative=False),
            make_request("d", prompt=384, output=2, speculative=False),
        ]
        batcher = LengthAwareTargetBatcher(
            short_threshold=1024,
            bucket_width=256,
            min_prefill_budget=1024,
            max_prefill_budget=1024,
        )
        sim = PhaseAwareSimulator(
            requests,
            PhaseConfig(max_num_seqs=4, max_num_batched_tokens=2048),
            self.profile,
            QueuePressurePhasePolicy(batcher),
        )
        plan = batcher.plan(sim)
        self.assertGreaterEqual(len(plan), 2)
        buckets = {
            (item.request.prompt_tokens - 1) // batcher.bucket_width
            for item in plan
        }
        self.assertEqual(len(buckets), 1)

    def test_length_aware_prefill_budget_adapts_to_backlog(self):
        batcher = LengthAwareTargetBatcher(
            min_prefill_budget=256,
            max_prefill_budget=1024,
        )
        small = PhaseAwareSimulator(
            [make_request("small", prompt=128, speculative=False)],
            PhaseConfig(),
            self.profile,
            QueuePressurePhasePolicy(batcher),
        )
        large = PhaseAwareSimulator(
            [
                make_request(f"large-{i}", prompt=4096, speculative=False)
                for i in range(8)
            ],
            PhaseConfig(),
            self.profile,
            QueuePressurePhasePolicy(batcher),
        )
        small_budget = batcher.prefill_budget_tokens(small, 2048)
        large_budget = batcher.prefill_budget_tokens(large, 2048)
        self.assertGreaterEqual(small_budget, batcher.min_prefill_budget)
        self.assertLessEqual(large_budget, batcher.max_prefill_budget)
        self.assertGreater(large_budget, small_budget)
        small.execute_action(PhaseAction.TARGET)
        self.assertEqual(len(small.adaptive_prefill_budget_samples), 1)
        self.assertGreater(small.adaptive_prefill_budget_samples[0], 0)

    def test_short_prefill_wait_window_is_adaptive(self):
        batcher = LengthAwareTargetBatcher(
            base_wait_ms=2.0,
            min_wait_ms=0.5,
            max_wait_ms=5.0,
        )
        sim = PhaseAwareSimulator(
            [make_request("short", prompt=128, speculative=False)],
            PhaseConfig(),
            self.profile,
            QueuePressurePhasePolicy(batcher),
        )
        sim.short_prefill_intervals_ms[:] = [4.0, 8.0, 12.0]
        self.assertEqual(batcher.adaptive_wait_window_ms(sim), 2.0)
        sim.short_prefill_intervals_ms[:] = [40.0, 60.0]
        self.assertEqual(batcher.adaptive_wait_window_ms(sim), 5.0)

    def test_joint_policy_waits_for_similar_short_prefills_only_when_draft_can_run(self):
        first = make_request("active", prompt=128, speculative=True)
        second = make_request("waiting", arrival=0.5, prompt=128, speculative=False)
        batcher = LengthAwareTargetBatcher(base_wait_ms=2.0)
        policy = JointPhasePolicy(batcher)
        sim = PhaseAwareSimulator(
            [first, second],
            PhaseConfig(confidence_threshold=0.0),
            self.profile,
            policy,
        )
        sim.prefill_queue.remove(first)
        first.status = RequestStatus.ACTIVE_DRAFT
        sim.active_draft.append(first)
        sim.now_ms = 0.5
        sim.refresh()
        self.assertTrue(batcher.should_wait_for_short_batch(sim))
        self.assertEqual(policy.choose_action(sim), PhaseAction.DRAFT)

        sim.now_ms = 3.0
        self.assertFalse(batcher.should_wait_for_short_batch(sim))

    def test_wait_window_counts_requests_in_the_same_length_bucket(self):
        requests = [
            make_request("active", prompt=128),
            make_request("bucket-a", arrival=0.1, prompt=128, speculative=False),
            make_request("bucket-b", arrival=0.2, prompt=384, speculative=False),
            make_request("bucket-a-2", arrival=0.3, prompt=256, speculative=False),
        ]
        batcher = LengthAwareTargetBatcher(
            short_threshold=512, bucket_width=256, base_wait_ms=2.0
        )
        policy = JointPhasePolicy(batcher)
        sim = PhaseAwareSimulator(
            requests, PhaseConfig(), self.profile, policy
        )
        active = sim.prefill_queue.popleft()
        active.status = RequestStatus.ACTIVE_DRAFT
        sim.active_draft.append(active)
        sim.now_ms = 0.2
        sim.refresh()
        self.assertEqual(len(sim.prefill_queue), 2)
        self.assertTrue(batcher.should_wait_for_short_batch(sim))

        sim.now_ms = 0.3
        sim.refresh()
        self.assertEqual(len(sim.prefill_queue), 3)
        self.assertFalse(batcher.should_wait_for_short_batch(sim))

    def test_ar_only_request_generates_one_token_per_target_step(self):
        req = make_request("ar", prompt=128, output=4, speculative=False)
        sim = PhaseAwareSimulator(
            [req],
            PhaseConfig(),
            self.profile,
            QueuePressurePhasePolicy(),
        )
        sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(req.committed_tokens, 1)
        self.assertEqual(req.status, RequestStatus.ACTIVE_AR)
        sim.execute_action(PhaseAction.TARGET)
        sim.execute_action(PhaseAction.TARGET)
        sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(req.committed_tokens, 4)
        self.assertEqual(req.status, RequestStatus.FINISHED)
        self.assertEqual(sim.ar_token_count, 3)

    def test_ttft_is_measured_at_prefill_completion(self):
        req = make_request("ttft", prompt=768, output=2, speculative=False)
        sim = PhaseAwareSimulator(
            [req],
            PhaseConfig(max_num_batched_tokens=512),
            self.profile,
            QueuePressurePhasePolicy(),
        )
        sim.execute_action(PhaseAction.TARGET)
        self.assertIsNone(req.first_token_time_ms)
        sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(req.first_token_time_ms, req.prefill_complete_time_ms)
        self.assertAlmostEqual(
            sim.ttft_samples_ms[0],
            req.prefill_complete_time_ms - req.arrival_time_ms,
        )

    def test_all_policies_complete_identical_mixed_workload(self):
        requests = generate_phase_workload(
            request_count=24,
            arrival_rate_rps=30.0,
            arrival_mode="bursty",
            seed=19,
            max_output_tokens=12,
            draft_caps=(2, 4),
            prompt_lengths=(128, 256, 512, 1024, 2048),
            ar_fraction=0.25,
        )
        config = PhaseConfig(
            max_num_seqs=4,
            max_num_batched_tokens=2048,
            max_draft_batch_size=4,
            fixed_h=4,
            confidence_threshold=0.0,
            max_steps=20_000,
            seed=19,
        )
        policies = [
            FixedHPhasePolicy(4, VllmChunkedTargetBatcher()),
            QueuePressurePhasePolicy(VllmChunkedTargetBatcher()),
            QueuePressurePhasePolicy(SarathiTargetBatcher(256)),
            QueuePressurePhasePolicy(LengthAwareTargetBatcher()),
            JointPhasePolicy(LengthAwareTargetBatcher()),
        ]
        expected_output = sum(req.max_output_tokens for req in requests)
        for policy in policies:
            with self.subTest(policy=policy.name):
                sim = PhaseAwareSimulator(requests, config, self.profile, policy)
                result = sim.run()
                self.assertEqual(result["completed_requests"], len(requests))
                self.assertEqual(result["output_tokens"], expected_output)
                self.assertGreater(result["target_forwards"], 0)
                self.assertGreater(result["ttft_p50_ms"], 0.0)
                self.assertLessEqual(result["target_slot_utilization"], 1.0)
                self.assertLessEqual(result["target_token_budget_utilization"], 1.0)

    def test_target_and_draft_forwards_are_serial(self):
        requests = generate_phase_workload(
            12, 20.0, arrival_mode="uniform", seed=7,
            max_output_tokens=8, draft_caps=(2, 4), ar_fraction=0.3,
        )
        sim = PhaseAwareSimulator(
            requests,
            PhaseConfig(confidence_threshold=0.0),
            self.profile,
            QueuePressurePhasePolicy(LengthAwareTargetBatcher()),
        )
        sim.run()
        previous_end = 0.0
        for record in sim.step_records:
            self.assertGreaterEqual(record.start_ms, previous_end)
            self.assertGreater(record.end_ms, record.start_ms)
            previous_end = record.end_ms

    def test_target_composition_reports_mixed_batches(self):
        requests = [
            make_request("spec", output=3, cap=1),
            make_request("ar", output=3, speculative=False),
            make_request("prefill", arrival=0.1, output=3, speculative=False),
        ]
        sim = PhaseAwareSimulator(
            requests,
            PhaseConfig(confidence_threshold=0.0),
            self.profile,
            FixedHPhasePolicy(1),
        )
        sim.execute_action(PhaseAction.TARGET)
        sim.execute_action(PhaseAction.DRAFT)
        record = sim.execute_action(PhaseAction.TARGET)
        self.assertEqual(record.verify_requests, 1)
        self.assertEqual(record.ar_requests, 1)
        self.assertEqual(record.prefill_requests, 1)


if __name__ == "__main__":
    unittest.main()
