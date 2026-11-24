import os
import unittest

from pr_check import check, smoke_blocks


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

    def test_runtime_build_inputs_require_smoke_until_approved(self):
        for path in ("crates/xcl/data/stubdb.toml", "crates/xcl/build.rs", "crates/xcl/Cargo.toml",
                     "Cargo.toml", "Cargo.lock"):
            with self.subTest(path=path):
                problems = check(BODY, HEAD, [changed(path)], lambda _: False)
                self.assertEqual(2, len(problems))
                self.assertTrue(all("Regression run" in problem for problem in problems))
                self.assertEqual([], check(BODY, HEAD, [changed(path)], lambda _: False, test_only=True))
        for path in ("tests/testtitle/Cargo.toml", "tests/testgamedll/build.rs", "crates/xcl/tests/data/x.toml"):
            with self.subTest(path=path):
                self.assertEqual([], check(BODY, HEAD, [changed(path)], lambda _: False))

    def test_test_paths_do_not_need_approval(self):
        files = [changed("crates/xcl/tests/integration.rs"), changed("crates/xcl/src/gpu/recomp/tests.rs")]
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False))


class SmokeBlockTests(unittest.TestCase):
    def test_two_blocks_in_one_fence_are_two_blocks(self):
        body = ("```\nxcl-smoke v1\ncommit: 1111111aaaa\ntitle: G\nseconds: 60\n\n"
                "xcl-smoke v1\ncommit: 1234567890\ntitle: G\nseconds: 60\n```\n")
        blocks = smoke_blocks(body)
        self.assertEqual(["1111111aaaa", "1234567890"], [b["commit"] for b in blocks])
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        self.assertEqual([], check(BODY + body, HEAD, files, lambda sha: sha == "1111111aaaa"))

    def test_blocks_in_separate_fences_stay_separate(self):
        body = ("```\nxcl-smoke v1\ncommit: 1111111aaaa\ntitle: G\n```\ntext\n"
                "```\nxcl-smoke v1\ncommit: 1234567890\ntitle: G\n```\n")
        self.assertEqual(["1111111aaaa", "1234567890"], [b["commit"] for b in smoke_blocks(body)])


class TemplateSectionTests(unittest.TestCase):
    TEMPLATE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 ".github", "pull_request_template.md"), encoding="utf-8").read()

    def problems(self, body):
        return check(body, HEAD, [], lambda _: True)

    def test_the_untouched_template_has_empty_sections(self):
        problems = self.problems(self.TEMPLATE)
        for section in ("Tested", "Not tested", "AI use"):
            self.assertIn(f"the **{section}** section is missing or empty", problems)

    def test_a_filled_template_passes(self):
        body = self.TEMPLATE.replace(
            "- hardware (GPU + driver, CPU), Rust version, debug / release:",
            "- RTX 3060 Ti, driver 581.xx, i7-13700K, release")
        body = body.replace(
            "- `cargo test --workspace -- --test-threads=4`: N passed, N failed, N ignored (SKIPPED GPU tests: …)",
            "- `cargo test --workspace -- --test-threads=4`: 676 passed, 0 failed, 41 ignored")
        body = body.replace("**Not tested**\n-\n", "**Not tested**\n- on a retail title\n")
        body = body.replace("- none / tool and model, and what it did",
                            "- Devin (Cognition), SWE-2: wrote the change and ran the tests")
        self.assertEqual([], self.problems(body))

    def test_comments_and_lone_bullets_do_not_count(self):
        body = ("**Tested**\n<!-- - hardware (GPU + driver, CPU), Rust version, debug / release: -->\n"
                "**Not tested**\n-\n**AI use**\n- none / tool and model, and what it did\n")
        problems = self.problems(body)
        self.assertIn("the **Tested** section is missing or empty", problems)
        self.assertIn("the **Not tested** section is missing or empty", problems)
        self.assertIn("the **AI use** section is missing or empty", problems)

    def test_ai_use_none_is_an_answer(self):
        body = ("**Tested**\nRTX 3060 Ti, release\n**Not tested**\n- nothing: docs only\n"
                "**AI use**\n- none\n")
        self.assertEqual([], self.problems(body))


if __name__ == "__main__":
    unittest.main()
