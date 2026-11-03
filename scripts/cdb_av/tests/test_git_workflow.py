import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "git_workflow.py"
SPEC = importlib.util.spec_from_file_location("git_workflow", MODULE_PATH)
assert SPEC and SPEC.loader
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class GitWorkflowTests(unittest.TestCase):
    def test_redact_https_credentials(self):
        self.assertEqual(
            workflow.redact_url("https://username:secret@github.com/team/repo.git"),
            "https://github.com/team/repo.git",
        )

    def test_keep_ssh_remote(self):
        self.assertEqual(
            workflow.redact_url("git@github.com:evian12/vllm.git"),
            "git@github.com:evian12/vllm.git",
        )

    def test_reject_model_and_raw_artifact_paths(self):
        for name in ("models/model.safetensors", "weights/foo.pt", "results/x/raw/data.json",
                     "results/x/server.log", ".venv/bin/python"):
            with self.subTest(name=name):
                self.assertIsNotNone(workflow.unsafe_path(name))

    def test_reject_secret_environment_paths(self):
        self.assertIsNotNone(workflow.unsafe_path(".env"))
        self.assertIsNotNone(workflow.unsafe_path(".env.production"))

    def test_allow_reproducibility_files(self):
        for name in ("AGENTS.md", "scripts/cdb_av/run_512_stress.sh",
                     "results/experiment/summary.csv", "results/experiment/summary.json",
                     ".env.example"):
            with self.subTest(name=name):
                self.assertIsNone(workflow.unsafe_path(name))

    def test_detect_private_key(self):
        sample = "-----BEGIN " + "PRIVATE KEY-----"
        self.assertIsNotNone(workflow.PRIVATE_CONTENT.search(sample))

    def test_detect_assigned_secret(self):
        sample = "api_" + 'key="' + "abcdefghijklmnop" + '"'
        self.assertIsNotNone(workflow.ASSIGNMENT_SECRET.search(sample))

    def test_count_test_results(self):
        self.assertEqual(workflow.count_tests("Ran 8 tests in 0.1s"), 8)
        self.assertEqual(workflow.count_tests("12 passed in 1.2s"), 12)


if __name__ == "__main__":
    unittest.main()
