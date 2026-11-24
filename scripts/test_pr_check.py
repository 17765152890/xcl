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

    def test_test_paths_do_not_need_approval(self):
        files = [changed("crates/xcl/tests/integration.rs"), changed("crates/xcl/src/gpu/recomp/tests.rs")]
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False))


def smoke(commit):
    return (f"xcl-smoke v1\ncommit: {commit}\ntitle: G\nseconds: 60\nframes shown: 100\n"
            "peak commit: 100 MB\ndevice removals: 0\ncrashes: 0\nexit: ok\n")


class SmokeBlockTests(unittest.TestCase):
    def test_two_blocks_in_one_fence_are_two_blocks(self):
        body = "```\n" + smoke("1111111aaaa") + "\n" + smoke(HEAD) + "```\n"
        blocks = smoke_blocks(body)
        self.assertEqual(["1111111aaaa", HEAD], [b.get("commit") for b in blocks])

    def test_two_blocks_in_one_fence_pass_the_regression_check(self):
        body = BODY + "**Regression run**\n```\n" + smoke("1111111aaaa") + "\n" + smoke(HEAD) + "```\n"
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        self.assertEqual([], check(body, HEAD, files, lambda sha: sha == "1111111aaaa"))


if __name__ == "__main__":
    unittest.main()
