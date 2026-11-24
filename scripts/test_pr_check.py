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


class BannedModelTests(unittest.TestCase):
    def body_for(self, model):
        return f"**Tested**\nSynthetic fixtures\n**Not tested**\nRetail titles\n**AI use**\n{model}\n"

    def test_phi_family_is_rejected_in_common_spellings(self):
        for model in ("Phi", "Phi 4", "Phi-4", "Phi4", "PHI 3.5"):
            with self.subTest(model=model):
                problems = check(self.body_for(model), HEAD, [], lambda _: False)
                self.assertTrue(any("does not accept" in problem for problem in problems), problems)

    def test_compact_banned_family_names_remain_rejected(self):
        for model in ("Qwen3", "Llama3", "Gemma2"):
            with self.subTest(model=model):
                problems = check(self.body_for(model), HEAD, [], lambda _: False)
                self.assertTrue(any("does not accept" in problem for problem in problems), problems)

    def test_words_starting_with_phi_are_not_rejected(self):
        self.assertEqual([], check(self.body_for("Philip"), HEAD, [], lambda _: False))

    def test_allowed_models_are_accepted(self):
        for model in ("None", "Claude Code", "GPT", "Gemini"):
            with self.subTest(model=model):
                self.assertEqual([], check(self.body_for(model), HEAD, [], lambda _: False))


if __name__ == "__main__":
    unittest.main()
