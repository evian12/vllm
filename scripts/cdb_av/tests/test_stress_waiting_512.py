# SPDX-License-Identifier: Apache-2.0
"""Protocol checks for the paired 512-request load generator."""

import io
import hashlib
import json
import unittest
from unittest.mock import patch

from scripts.cdb_av import stress_waiting_512 as client


class StressClientProtocolTests(unittest.TestCase):
    def test_dry_run_carries_stable_request_id_and_k_cap(self):
        row = client.send_request(3, 18084, 32, "paired", dry_run=True)
        self.assertEqual(row["payload"]["request_id"], "p0_paired_3")
        self.assertEqual(row["payload"]["vllm_xargs"], {
            "cdb_draft_k": 8,
            "cdb_trace_key": "stress512:3",
        })
        self.assertEqual(row["requested_K"], 8)
        self.assertIsNone(row["effective_K"])
        self.assertIsNone(row["actual_draft_steps"])

    def test_http_success_requires_completion_body(self):
        invalid = io.BytesIO(json.dumps({"error": "engine died"}).encode())
        with patch.object(client.urllib.request, "urlopen", return_value=invalid):
            row = client.send_request(0, 18084, 32, "paired")
        self.assertFalse(row["ok"])
        self.assertIn("Invalid completion response", row["error"])

    def test_valid_completion_records_server_usage(self):
        response = {
            "id": "cmpl-p0_paired_0",
            "choices": [{"text": "answer", "finish_reason": "length"}],
            "usage": {"completion_tokens": 32},
        }
        with patch.object(
            client.urllib.request,
            "urlopen",
            return_value=io.BytesIO(json.dumps(response).encode()),
        ):
            row = client.send_request(0, 18084, 32, "paired")
        self.assertTrue(row["ok"])
        self.assertEqual(row["output_tokens"], 32)
        self.assertEqual(row["output_sha256"], hashlib.sha256(b"answer").hexdigest())
        self.assertEqual(row["response_id"], response["id"])
        self.assertLessEqual(row["started_at"], row["completed_at"])

    def test_opt_in_exact_token_ids_and_manual_cap(self):
        response = {
            "id": "cmpl-p0_reference_0",
            "choices": [{"text": "answer", "finish_reason": "length",
                         "prompt_token_ids": [11, 12], "token_ids": [13, 14]}],
            "usage": {"completion_tokens": 2},
        }
        with patch.object(client.urllib.request, "urlopen",
                          return_value=io.BytesIO(json.dumps(response).encode())):
            row = client.send_request(0, 18084, 2, "reference", False, 4, True)
        self.assertTrue(row["ok"])
        self.assertEqual(row["requested_K"], 4)
        self.assertEqual(row["prompt_token_ids"], [11, 12])
        self.assertEqual(row["generated_token_ids"], [13, 14])

    def test_missing_exact_token_ids_fails_trace_collection(self):
        response = {
            "id": "cmpl-p0_reference_0",
            "choices": [{"text": "answer", "finish_reason": "length"}],
            "usage": {"completion_tokens": 2},
        }
        with patch.object(client.urllib.request, "urlopen",
                          return_value=io.BytesIO(json.dumps(response).encode())):
            row = client.send_request(0, 18084, 2, "reference", False, 4, True)
        self.assertFalse(row["ok"])
        self.assertIn("token-ID evidence", row["error"])


if __name__ == "__main__":
    unittest.main()
