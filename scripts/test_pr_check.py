from pathlib import Path
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


class TemplatePlaceholderTests(unittest.TestCase):
    def test_untouched_pr_template_fails_all_required_sections(self):
        template = Path(__file__).resolve().parent.parent / ".github" / "pull_request_template.md"
        body = template.read_text(encoding="utf-8")
        problems = check(body, HEAD, [changed("README.md")], lambda _: True)
        self.assertEqual(
            [
                "the **Tested** section is missing or empty",
                "the **Not tested** section is missing or empty",
                "the **AI use** section is missing or empty",
            ],
            problems,
        )

    def test_filled_template_with_comments_and_real_lines_passes(self):
        body = (
            "**Tested**\n"
            "<!-- comment only does not count, real line below does -->\n"
            "- hardware (GPU + driver, CPU), Rust version, debug / release: RTX 3070 Ti, Rust 1.88.0\n"
            "- `cargo test --workspace -- --test-threads=4`: N passed, N failed, N ignored (SKIPPED GPU tests: …)\n"
            "**Not tested**\n"
            "- AMD GPUs\n"
            "**AI use**\n"
            "- none\n"
        )
        self.assertEqual([], check(body, HEAD, [changed("README.md")], lambda _: True))

    def test_bare_bullets_and_html_comments_only_fail(self):
        body = "**Tested**\n<!-- todo -->\n-\n**Not tested**\n*\n**AI use**\n  -  \n"
        problems = check(body, HEAD, [changed("README.md")], lambda _: True)
        self.assertEqual(3, len(problems))


class SectionHeadingTests(unittest.TestCase):
    """Where a required section starts and ends, and what counts as its text."""

    FILLED = ("A change to a script.\n\n**What is in it**\n- `scripts/example.py`: the change\n\n"
              "**Tested**\n- `python3 scripts/example.py`: ok\n\n**Not tested**\n- Windows\n\n"
              "**AI use**\n- none\n")

    def problems(self, body):
        return check(body, HEAD, [changed("scripts/example.py")], lambda _: False)

    @staticmethod
    def empty(section):
        return f"the **{section}** section is missing or empty"

    def test_filled_description_passes(self):
        self.assertEqual([], self.problems(self.FILLED))
        self.assertEqual([], self.problems(self.FILLED.replace("\n", "\r\n")))

    def test_section_inside_a_comment_is_not_there(self):
        # the heading and its only result, both commented out
        body = self.FILLED.replace("**Tested**\n- `python3 scripts/example.py`: ok\n",
                                   "<!--\n**Tested**\n- `python3 scripts/example.py`: ok\n-->\n")
        self.assertEqual([self.empty("Tested")], self.problems(body))
        # the heading is there, its only result is commented out across lines
        body = self.FILLED.replace("- Windows\n", "<!--\n- Windows\n-->\n")
        self.assertEqual([self.empty("Not tested")], self.problems(body))

    def test_unclosed_comment_hides_what_follows(self):
        # GitHub shows nothing after an unclosed comment: the sections after it are not there
        body = self.FILLED.replace("- `python3 scripts/example.py`: ok\n", "<!-- results to come\n")
        self.assertEqual([self.empty(s) for s in ("Tested", "Not tested", "AI use")], self.problems(body))

    def test_heading_is_a_line_of_its_own(self):
        # a section name in bold at the end of a sentence is not the section's heading
        body = self.FILLED.replace("**Tested**\n- `python3 scripts/example.py`: ok\n",
                                   "Nothing here was **Tested**\nas far as I know.\n")
        self.assertEqual([self.empty("Tested")], self.problems(body))
        # indented, or with spaces after it, it still is
        self.assertEqual([], self.problems(self.FILLED.replace("**Tested**\n", "  **Tested**  \n")))

    def test_heading_name_at_the_start_of_a_sentence_is_text(self):
        # only a heading on a line of its own ends the section before it
        body = self.FILLED.replace("- `python3 scripts/example.py`: ok\n",
                                   "**AI use** below covers the test too; 3 passed\n")
        self.assertEqual([], self.problems(body))

    def test_text_in_bold_is_text(self):
        # a line in bold that is not a heading of the outline does not end the section
        body = self.FILLED.replace("- Windows\n", "**AMD**: not run\n")
        self.assertEqual([], self.problems(body))

    def test_empty_section_before_another_one(self):
        body = self.FILLED.replace("- `python3 scripts/example.py`: ok\n", "")
        self.assertEqual([self.empty("Tested")], self.problems(body))
        self.assertEqual([self.empty("AI use")], self.problems(self.FILLED.replace("- none\n", "")))


class BannedModelTests(unittest.TestCase):
    """The model check reads what is visible under **AI use**, as the section check does."""

    FILLED = SectionHeadingTests.FILLED
    QWEN = FILLED.replace("- none\n", "- Qwen3-Coder wrote the code\n")

    def problems(self, body):
        return check(body, HEAD, [changed("scripts/example.py")], lambda _: False)

    @staticmethod
    def banned(*models):
        return ("**AI use** names a model this repository does not accept for code ("
                + ", ".join(models) + "): see CONTRIBUTING.md")

    def test_model_under_ai_use_is_refused(self):
        self.assertEqual([self.banned("qwen")], self.problems(self.QWEN))
        self.assertEqual([], self.problems(self.FILLED))

    def test_section_name_elsewhere_does_not_hide_the_section(self):
        # in a comment at the top, or in a sentence of another section
        body = "<!-- see **AI use** at the end -->\n" + self.QWEN
        self.assertEqual([self.banned("qwen")], self.problems(body))
        body = self.QWEN.replace("- `scripts/example.py`: the change\n",
                                 "- the **AI use** section is short\n")
        self.assertEqual([self.banned("qwen")], self.problems(body))

    def test_bold_line_does_not_end_the_section(self):
        body = self.FILLED.replace("- none\n", "- Claude Code\n**also** Qwen3 for the tests\n")
        self.assertEqual([self.banned("qwen")], self.problems(body))

    def test_comment_is_not_what_was_used(self):
        body = self.FILLED.replace("- none\n", "<!-- not qwen or llama -->\n- Claude Code\n")
        self.assertEqual([], self.problems(body))

    def test_model_in_another_section_is_not_ai_use(self):
        body = self.FILLED.replace("- Windows\n", "- nothing with Llama\n")
        self.assertEqual([], self.problems(body))


if __name__ == "__main__":
    unittest.main()
