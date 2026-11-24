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


MAIN = "abcdef1234567890"


def smoke(commit, *, title="SampleTitle", seconds=60, exit_code="0x0", frames="1000 (PresentX 1000)",
          removals=0, crashes=0, peak="1000 MB"):
    return (
        "```\n"
        "xcl-smoke v1\n"
        f"commit: {commit}\n"
        "gpu: Test GPU\n"
        f"title: {title}\n"
        f"seconds: {seconds}\n"
        f"exit: {exit_code}\n"
        f"frames shown: {frames}\n"
        "presents per 2 s (last 3): 60 60 60\n"
        f"device removals: {removals}\n"
        f"crashes: {crashes}\n"
        f"peak commit: {peak}\n"
        "```\n"
    )


def body_with_smoke(main_block, pr_block):
    return f"{BODY}\n**Regression run**\n\n{main_block}\n{pr_block}\n"


class PrCheckRulesTests(unittest.TestCase):
    def test_missing_or_empty_sections_are_reported(self):
        body = "**Tested**\n   \n**Not tested**\n\n**AI use**\nNone\n"
        problems = check(body, HEAD, [], lambda _: False)
        self.assertEqual(2, len(problems), problems)
        self.assertTrue(any("**Tested**" in p for p in problems), problems)
        self.assertTrue(any("**Not tested**" in p for p in problems), problems)

        adjacent = "**Tested**\n**Regression run**\nnone\n**Not tested**\n**AI use**\n   \n"
        problems = check(adjacent, HEAD, [], lambda _: False)
        self.assertEqual(3, len(problems), problems)

        subheading = "**Tested**\n\n**Machine:** RTX 5070\n**Not tested**\nnone\n**AI use**\nNone\n"
        self.assertEqual([], check(subheading, HEAD, [], lambda _: False))

    def test_banned_models_are_rejected_and_frontier_models_accepted(self):
        for model in ("Qwen3", "DeepSeek-R1", "Llama-4", "Mistral", "Mixtral", "Gemma", "Phi-4", "phi3"):
            body = f"**Tested**\nok\n**Not tested**\nnone\n**AI use**\nUsed {model}\n"
            problems = check(body, HEAD, [], lambda _: False)
            self.assertEqual(1, len(problems), (model, problems))
            self.assertIn("AI use", problems[0])
        ok_body = "**Tested**\nok\n**Not tested**\nnone\n**AI use**\nClaude Opus 5.5\n"
        self.assertEqual([], check(ok_body, HEAD, [], lambda _: False))

    def test_game_files_binaries_test_data_limit_and_removed_files(self):
        files = [
            ("tests/data/testgame/MicrosoftGame.config", "modified", False, lambda: 100),
            ("tests/data/small.bin", "added", True, lambda: 1 << 20),
            ("old/deleted.rpf", "removed", True, lambda: 10 << 20),
        ]
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False))

        bad_files = [
            ("game/MicrosoftGame.config", "added", False, lambda: 100),
            ("appxmanifest.xml", "added", False, lambda: 100),
            ("dir/resources.pri", "added", False, lambda: 100),
            ("tests/data/too_big.bin", "added", True, lambda: (1 << 20) + 1),
            ("crates/xcl/tests/blob.dds", "added", False, lambda: 64),
            ("docs/pic.png", "added", True, lambda: 64),
        ]
        problems = check(BODY, HEAD, bad_files, lambda _: False)
        self.assertEqual(len(bad_files), len(problems), problems)

    def test_valid_smoke_pair_and_exact_thresholds_pass(self):
        runtime_files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        body = body_with_smoke(
            smoke(MAIN, frames="1000 (PresentX 1000)", peak="1000 MB"),
            smoke(HEAD[:12], frames="800 (PresentX 800)", peak="1250 MB"),
        )
        self.assertEqual([], check(body, HEAD, runtime_files, lambda sha: MAIN.startswith(sha)))

    def test_uncommitted_pr_smoke_and_non_main_base_are_rejected(self):
        runtime_files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        body = body_with_smoke(
            smoke("deadbeefcafe"),
            smoke(f"{HEAD} (uncommitted changes)"),
        )
        problems = check(body, HEAD, runtime_files, lambda sha: MAIN.startswith(sha))
        self.assertEqual(2, len(problems), problems)
        self.assertTrue(any("no uncommitted changes" in p for p in problems), problems)
        self.assertTrue(any("commit of main" in p for p in problems), problems)

    def test_smoke_rejects_removals_crashes_killed_and_mismatched_title_or_seconds(self):
        runtime_files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        body = body_with_smoke(
            smoke(MAIN, title="TitleA", seconds=60),
            smoke(HEAD, title="TitleB", seconds=30, exit_code="killed (no exit)", removals=1, crashes=2),
        )
        problems = check(body, HEAD, runtime_files, lambda sha: MAIN.startswith(sha))
        for expected in ("device removals", "crashes", "killed", "same title for the same time"):
            self.assertTrue(any(expected in p for p in problems), (expected, problems))

    def test_smoke_rejects_frame_drop_and_commit_growth(self):
        runtime_files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        body = body_with_smoke(
            smoke(MAIN, frames="1000 (PresentX 1000)", peak="1000 MB"),
            smoke(HEAD, frames="799 (PresentX 799)", peak="1251 MB"),
        )
        problems = check(body, HEAD, runtime_files, lambda sha: MAIN.startswith(sha))
        self.assertEqual(2, len(problems), problems)
        self.assertTrue(any("fewer frames" in p for p in problems), problems)
        self.assertTrue(any("more memory" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
