import os
import subprocess
import sys
import tempfile
import unittest

from pr_check import check, looks_binary

HERE = os.path.dirname(os.path.abspath(__file__))


HEAD = "1234567890abcdef"
BODY = "**Tested**\nSynthetic fixtures\n**Not tested**\nRetail titles\n**AI use**\nNone\n"


def changed(path, binary=False):
    return (path, "modified", binary, lambda: 10)


class TestOnlyApprovalTests(unittest.TestCase):
    def test_runtime_path_requires_smoke_until_approved(self):
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        problems = check(BODY, HEAD, files, lambda _: False)
        self.assertEqual(2, len(problems))
        self.assertTrue(all("Regression run" in problem for problem in problems))
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False, test_only=True))

    def test_approval_keeps_description_and_asset_checks(self):
        files = [
            changed("crates/xcl/src/hle/d3d12x/vram.rs"),
            changed("assets/content.rpf"),
            changed("notes/attachment.txt", binary=True),
        ]
        body = "**Tested**\nSynthetic fixtures\n**AI use**\nQwen\n"
        problems = check(body, HEAD, files, lambda _: False, test_only=True)
        self.assertEqual(4, len(problems))
        for reason in ("Not tested", "AI use", "content.rpf", "attachment.txt"):
            self.assertTrue(any(reason in problem for problem in problems), problems)

    def test_test_paths_do_not_need_approval(self):
        files = [changed("crates/xcl/tests/integration.rs"), changed("crates/xcl/src/gpu/recomp/tests.rs")]
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False))


def blob(data):
    """Store `data` as a blob in this repository's object database; returns its sha."""
    r = subprocess.run(["git", "hash-object", "-w", "--stdin"], input=data, capture_output=True, cwd=HERE, check=True)
    return r.stdout.decode().strip()


def run_local(rows):
    """pr_check.py --local on a files.tsv of `rows` (path, status, the workflow's binary flag, content)."""
    with tempfile.TemporaryDirectory() as tmp:
        body, files = os.path.join(tmp, "body.md"), os.path.join(tmp, "files.tsv")
        with open(body, "w", encoding="utf-8") as f:
            f.write(BODY)
        with open(files, "w", encoding="utf-8") as f:
            for path, status, flag, data in rows:
                f.write(f"{path}\t{status}\t{'true' if flag else 'false'}\t{blob(data)}\n")
        r = subprocess.run([sys.executable, os.path.join(HERE, "pr_check.py"), "--body", body, "--head", HEAD,
                            "--files", files, "--local"], capture_output=True, text=True, cwd=HERE)
        return r.returncode, r.stdout


class BinaryContentTests(unittest.TestCase):
    # The workflow marks a file binary when GitHub shows no patch and 0 changed lines, which is also true
    # of an empty file, a pure rename and a mode change; pr_check.py confirms with the content.
    def test_looks_binary(self):
        self.assertFalse(looks_binary(b""))
        self.assertFalse(looks_binary(b"fn main() {}\n"))
        self.assertTrue(looks_binary(b"MZ\x90\x00\x03\x00"))
        self.assertFalse(looks_binary(b"x" * 8000 + b"\x00"))   # git looks at 8000 bytes only

    def test_an_empty_file_and_a_pure_rename_are_not_binary(self):
        code, out = run_local([("scripts/re/.gitkeep", "added", True, b""),
                               ("scripts/re/pe_exports_moved.py", "renamed", True, b"import sys\nprint(1)\n")])
        self.assertEqual((code, out.strip()), (0, "description ok"))

    def test_a_real_binary_is_still_refused(self):
        for path, status in (("docs/picture.png", "added"), ("docs/moved.bin2", "renamed")):
            with self.subTest(path=path):
                code, out = run_local([(path, status, True, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")])
                self.assertEqual(code, 1)
                self.assertIn(f"`{path}` is a binary file", out)


if __name__ == "__main__":
    unittest.main()
