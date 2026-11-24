import unittest

from pr_check import check


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


def smoke_body(pr_lit, main_lit, pr_line=True, main_line=True):
    def block(commit, lit, include):
        line = f"\nlit at end: {lit}" if include else ""
        return (
            "```\n"
            "xcl-smoke v1\n"
            f"commit: {commit}\n"
            "gpu: test\n"
            "title: game\n"
            "seconds: 20\n"
            "exit: 0x0\n"
            "frames shown: 100 (PresentX 100)\n"
            "device removals: 0\n"
            "crashes: 0\n"
            f"peak commit: 1000 MB{line}\n"
            "```\n"
        )

    return BODY + "\n**Regression run**\n" + block("abcdef1234567890", main_lit, main_line) + block(HEAD, pr_lit, pr_line)


class LitFrameTests(unittest.TestCase):
    def problems(self, body):
        return check(body, HEAD, [changed("crates/xcl/src/hle/d3d12x/present.rs")], lambda sha: sha.startswith("abcdef"))

    def test_a_dark_frame_against_a_lit_main_is_refused(self):
        problems = self.problems(smoke_body("0 %", "80 %"))
        self.assertEqual(1, len(problems))
        self.assertIn("dark", problems[0])

    def test_a_fraction_under_one_percent_is_dark(self):
        problems = self.problems(smoke_body("0.4 %", "60 %"))
        self.assertEqual(1, len(problems))

    def test_a_lit_frame_passes(self):
        self.assertEqual([], self.problems(smoke_body("90 %", "80 %")))

    def test_missing_or_unsampled_lines_are_left_alone(self):
        self.assertEqual([], self.problems(smoke_body("0 %", "80 %", pr_line=False)))
        self.assertEqual([], self.problems(smoke_body("n/a", "80 %")))
        self.assertEqual([], self.problems(smoke_body("0 %", "40 %")))


if __name__ == "__main__":
    unittest.main()
