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

    def test_non_hex_main_commits_are_refused_before_ancestor_check(self):
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        bad_commits = (
            "refs/heads/main",
            "origin/main",
            "main~00",
            "HEAD~000",
            "47efa2e~0",
            "../../pulls",
            "main?per_page=1",
            "--version",
            "123456",
            "a" * 41,
        )
        for bad in bad_commits:
            with self.subTest(commit=bad):
                seen = []
                body = (
                    f"{BODY}```\nxcl-smoke v1\ncommit: {bad}\ntitle: G\nseconds: 60\n```\n"
                    f"```\nxcl-smoke v1\ncommit: {HEAD}\ntitle: G\nseconds: 60\n```\n"
                )
                problems = check(body, HEAD, files, lambda sha, seen=seen: (seen.append(sha), True)[1])
                self.assertEqual([], seen)
                self.assertEqual(1, len(problems))
                self.assertIn("commit of main", problems[0])

    def test_short_or_non_hex_head_commits_are_refused(self):
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        for bad in ("1 some other", "123456", "1234567~0", "refs/heads/pr"):
            with self.subTest(commit=bad):
                body = (
                    f"{BODY}```\nxcl-smoke v1\ncommit: 1111111aaaa\ntitle: G\nseconds: 60\n```\n"
                    f"```\nxcl-smoke v1\ncommit: {bad}\ntitle: G\nseconds: 60\n```\n"
                )
                problems = check(body, HEAD, files, lambda sha: sha == "1111111aaaa")
                self.assertEqual(1, len(problems))
                self.assertIn("head commit", problems[0])

    def test_valid_hex_commits_pass_case_insensitively(self):
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        full_main = "a" * 40
        full_head = "1234567890abcdef" + "0" * 24
        for main_sha, pr_sha in (
            ("1111111", full_head[:7]),
            ("1111111AAAA", full_head[:12].upper()),
            (full_main.upper(), full_head.upper()),
        ):
            with self.subTest(main_sha=main_sha, pr_sha=pr_sha):
                seen = []
                body = (
                    f"{BODY}```\nxcl-smoke v1\ncommit: {main_sha}\ntitle: G\nseconds: 60\n```\n"
                    f"```\nxcl-smoke v1\ncommit: {pr_sha}\ntitle: G\nseconds: 60\n```\n"
                )
                problems = check(body, full_head, files, lambda sha, seen=seen: (seen.append(sha), sha == main_sha.lower())[1])
                self.assertEqual([main_sha.lower()], seen)
                self.assertEqual([], problems)


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


if __name__ == "__main__":
    unittest.main()
