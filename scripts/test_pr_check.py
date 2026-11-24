import unittest

from pr_check import check, runtime


HEAD = "1234567890abcdef"
BODY = "**Tested**\nSynthetic fixtures\n**Not tested**\nRetail titles\n**AI use**\nNone\n"


def changed(path, binary=False):
    return (path, "modified", binary, lambda: 10)


class RuntimePathTests(unittest.TestCase):
    def test_test_files_are_not_runtime(self):
        for path in ("crates/xcl/src/gpu/recomp/tests.rs", "crates/xcl/src/gpu/adapter_tests.rs",
                     "crates/xcl/src/gpu/agility_tests.rs", "crates/xcl/src/hle/tests/vram.rs"):
            with self.subTest(path=path):
                self.assertFalse(runtime(path))

    def test_names_that_only_contain_tests_are_runtime(self):
        for path in ("crates/xcl/src/gpu/rdna2.rs", "crates/xcl/src/net/requests.rs", "crates/xcl/src/hle/contests.rs",
                     "crates/xcl/src/hle/testsupport/mod.rs", "crates/xcl/src/gpu/tests_layout.rs"):
            with self.subTest(path=path):
                self.assertTrue(runtime(path))

    def test_outside_the_runtime_crate(self):
        for path in ("README.md", "scripts/pr_check.py", "tests/testtitle/src/main.rs", "crates/xcl/tests/e2e.rs"):
            with self.subTest(path=path):
                self.assertFalse(runtime(path))

    def test_a_runtime_file_named_like_a_test_still_needs_a_regression_run(self):
        problems = check(BODY, HEAD, [changed("crates/xcl/src/hle/contests.rs")], lambda _: False)
        self.assertTrue(problems and all("Regression run" in problem for problem in problems), problems)


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


if __name__ == "__main__":
    unittest.main()
