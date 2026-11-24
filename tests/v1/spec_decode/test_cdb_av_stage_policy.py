# SPDX-License-Identifier: Apache-2.0
import unittest
from dataclasses import replace

from vllm.v1.spec_decode.cdb_av_stage_policy import StageState, choose_stage


class TestStagePolicy(unittest.TestCase):
    def setUp(self):
        self.state = StageState(
            now=1.0, target_deadlines=(1.1,), target_count=1,
            target_capacity=4, draft_can_progress=True,
            draft_step_time=0.005, target_time=0.01,
            draft_can_add_ready=True,
        )

    def test_slack_allows_draft(self):
        self.assertEqual(choose_stage(self.state), "draft")

    def test_urgent_target(self):
        self.assertEqual(choose_stage(replace(
            self.state, target_deadlines=(1.012,))), "target")

    def test_full_target(self):
        self.assertEqual(choose_stage(replace(
            self.state, target_count=4,
            target_deadlines=(1.1,) * 4)), "target")

    def test_no_productive_draft(self):
        self.assertEqual(choose_stage(replace(
            self.state, draft_can_add_ready=False)), "target")

    def test_draft_only(self):
        self.assertEqual(choose_stage(replace(
            self.state, target_count=0, target_deadlines=())), "draft")

    def test_target_only(self):
        self.assertEqual(choose_stage(replace(
            self.state, draft_can_progress=False)), "target")

    def test_idle(self):
        self.assertEqual(choose_stage(replace(
            self.state, target_count=0, target_deadlines=(),
            draft_can_progress=False)), "idle")

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            choose_stage(replace(self.state, target_count=-1))
        with self.assertRaises(ValueError):
            choose_stage(replace(self.state, target_deadlines=()))
        with self.assertRaises(ValueError):
            choose_stage(replace(self.state, draft_step_time=-1))
        with self.assertRaises(ValueError):
            choose_stage(replace(self.state, target_time=float("nan")))


if __name__ == "__main__":
    unittest.main()
