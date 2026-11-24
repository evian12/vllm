# SPDX-License-Identifier: Apache-2.0
"""Regression for eligible Target queues larger than a single batch."""
import unittest

from vllm.v1.spec_decode.cdb_av_stage_policy import StageState, choose_stage


class TestTargetBacklog(unittest.TestCase):
    def test_backlog_must_trigger_target(self):
        state = StageState(
            now=0.0, target_deadlines=(100.0,) * 12,
            target_count=12, target_capacity=8,
            draft_can_progress=True, draft_step_time=0.005,
            target_time=0.01, draft_can_add_ready=True,
        )
        self.assertEqual(choose_stage(state), "target")
