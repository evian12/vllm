"""Checkpoint integration tests using real temporary repositories and remotes."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "git_workflow.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_path_workflow", MODULE_PATH)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class CheckpointPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "work"
        remote = Path(self.temp.name) / "remote.git"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
        self.git("init", "-q", "-b", "codex/checkpoint-test")
        self.git("config", "user.name", "Checkpoint Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        for name in ("remove.txt", "other.txt"):
            (self.root / name).write_text("original\n")
        self.git("add", "--", "remove.txt", "other.txt")
        self.git("commit", "-qm", "fixture")
        self.git("remote", "add", "myfork", str(remote))
        self.git("push", "-q", "myfork", "HEAD:refs/heads/codex/checkpoint-test")
        self.original_head = self.git("rev-parse", "HEAD")
        self.patcher = patch.object(workflow, "ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def git(self, *args, check=True):
        result = subprocess.run(["git", *args], cwd=self.root, text=True,
                                capture_output=True, check=check)
        return result.stdout.strip()

    def validate_staged(self, paths):
        # Keep the real content guards. Avoid recursively invoking this suite
        # from a fixture repository that does not contain workflow scripts.
        errors = workflow.scan_staged()
        if errors:
            raise RuntimeError("Staged content rejected: " + "; ".join(errors))
        return {"status": "pass", "scope": "real staged-content guards"}

    def checkpoint(self, *paths):
        args = argparse.Namespace(path=list(paths), remote="myfork",
                                  message="fix: checkpoint fixture", pytest=[])
        with patch.object(workflow, "validate", side_effect=self.validate_staged):
            return workflow.checkpoint(args)

    def assert_pushed(self, report):
        self.assertNotEqual(report["commit"], self.original_head)
        self.assertEqual(report["commit"], report["remote_head"])
        self.assertEqual(self.git("status", "--porcelain"), "")

    def staged_delete(self):
        (self.root / "remove.txt").unlink()
        self.git("add", "--", "remove.txt")

    def test_already_staged_deletion_can_be_checkpointed(self):
        self.staged_delete()
        self.assert_pushed(self.checkpoint("remove.txt"))
        self.assertNotIn("remove.txt", self.git("ls-tree", "--name-only", "HEAD"))

    def test_staged_deletion_and_other_explicit_change(self):
        self.staged_delete()
        (self.root / "other.txt").write_text("updated\n")
        self.assert_pushed(self.checkpoint("remove.txt", "other.txt"))
        self.assertEqual(self.git("show", "HEAD:other.txt"), "updated")

    def test_unstaged_deletion_is_still_added(self):
        (self.root / "remove.txt").unlink()
        self.assert_pushed(self.checkpoint("remove.txt"))

    def test_recreated_file_is_added_over_staged_deletion(self):
        self.staged_delete()
        (self.root / "remove.txt").write_text("recreated\n")
        self.assert_pushed(self.checkpoint("remove.txt"))
        self.assertEqual(self.git("show", "HEAD:remove.txt"), "recreated")

    def test_deleted_staged_addition_does_not_commit_stale_blob(self):
        new = self.root / "new.txt"
        new.write_text("never commit this stale blob\n")
        self.git("add", "--", "new.txt")
        new.unlink()
        (self.root / "other.txt").write_text("updated\n")
        self.assert_pushed(self.checkpoint("new.txt", "other.txt"))
        self.assertNotIn("new.txt", self.git("ls-tree", "--name-only", "HEAD"))

    def test_dangling_symlink_recreation_reaches_content_guard(self):
        self.staged_delete()
        (self.root / "remove.txt").symlink_to("missing-target")
        with self.assertRaisesRegex(RuntimeError, "symlink requires explicit review"):
            self.checkpoint("remove.txt")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.original_head)
        self.assertTrue((self.root / "remove.txt").is_symlink())
        self.assertTrue(self.git("ls-files", "--stage", "remove.txt").startswith("120000"))

    def test_staged_symlink_guard_uses_index_even_after_worktree_replacement(self):
        path = self.root / "remove.txt"
        path.unlink()
        path.symlink_to("missing-target")
        self.git("add", "--", "remove.txt")
        path.unlink()
        path.write_text("ordinary worktree file\n")
        errors = workflow.scan_staged()
        self.assertTrue(any("symlink requires explicit review" in e for e in errors), errors)
        self.assertEqual(path.read_text(), "ordinary worktree file\n")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.original_head)

    def test_type_change_to_regular_file_runs_secret_guard(self):
        path = self.root / "remove.txt"
        path.unlink()
        path.symlink_to("missing-target")
        self.git("add", "--", "remove.txt")
        self.git("commit", "-qm", "tracked symlink fixture")
        self.git("push", "-q", "myfork", "HEAD:refs/heads/codex/checkpoint-test")
        self.original_head = self.git("rev-parse", "HEAD")
        path.unlink()
        path.write_text("api_" + 'key="' + "abcdefghijklmnop" + '"\n')
        with self.assertRaisesRegex(RuntimeError, "possible secret"):
            self.checkpoint("remove.txt")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.original_head)

    def test_unknown_missing_path_still_fails(self):
        self.staged_delete()
        with self.assertRaises(RuntimeError):
            self.checkpoint("remove.txt", "typo.txt")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.original_head)
        self.assertEqual(workflow.remote_sha("myfork", "codex/checkpoint-test"), self.original_head)

    def test_unrelated_staged_change_is_not_committed(self):
        self.staged_delete()
        (self.root / "other.txt").write_text("unrelated\n")
        self.git("add", "--", "other.txt")
        with self.assertRaisesRegex(RuntimeError, "Unrelated paths"):
            self.checkpoint("remove.txt")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.original_head)
        self.assertIn("other.txt", self.git("diff", "--cached", "--name-only"))

    def test_unsafe_explicit_path_remains_blocked(self):
        self.staged_delete()
        with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
            self.checkpoint("remove.txt", "../outside.txt")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.original_head)

    def test_staged_rename_with_both_paths_is_checkpointed(self):
        self.git("mv", "remove.txt", "renamed.txt")
        self.assert_pushed(self.checkpoint("remove.txt", "renamed.txt"))
        self.assertEqual(self.git("show", "HEAD:renamed.txt"), "original")

    def test_staged_rename_requires_old_path_to_be_explicit(self):
        self.git("mv", "remove.txt", "renamed.txt")
        with self.assertRaisesRegex(RuntimeError, "Unrelated paths"):
            self.checkpoint("renamed.txt")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.original_head)

    def test_duplicate_explicit_path_preserves_staged_deletion(self):
        self.staged_delete()
        self.assert_pushed(self.checkpoint("remove.txt", "remove.txt"))

    def test_non_ascii_and_newline_deleted_paths_are_not_git_quoted_names(self):
        for name in ("删除.txt", "line\nbreak.txt", " leading.txt"):
            with self.subTest(name=name):
                path = self.root / name
                path.write_text("fixture\n")
                self.git("add", "--", name)
                self.git("commit", "-qm", "special path fixture")
                self.git("push", "-q", "myfork", "HEAD:refs/heads/codex/checkpoint-test")
                self.original_head = self.git("rev-parse", "HEAD")
                path.unlink()
                self.git("add", "--", name)
                self.assert_pushed(self.checkpoint(name))


if __name__ == "__main__":
    unittest.main()
