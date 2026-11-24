import unittest
from pathlib import Path
from runpy import run_path

check = run_path(str(Path(__file__).with_name("pr_check.py")))["check"]


def problems(pr_exit, main_exit):
    def block(commit, status):
        return (f"```\nxcl-smoke v1\ncommit: {commit}\ntitle: synthetic\nseconds: 60\n"
                f"exit: {status}\ncrashes: 0\ndevice removals: 0\n```\n")

    main, head = "a" * 40, "b" * 40
    body = ("**Tested**\nsynthetic\n**Not tested**\nretail\n**AI use**\ntest fixture\n"
            + block(main, main_exit) + block(head, pr_exit))
    return check(body, head, [("crates/xcl/src/main.rs", "modified", False, lambda: 0)],
                 lambda commit: commit == main)


class SmokeExitTest(unittest.TestCase):
    def test_abnormal_pr_exit_without_logged_crash(self):
        self.assertTrue(any("0xc0000005" in p for p in problems("0xc0000005", "0x0")))

    def test_fix_can_have_a_crashing_baseline(self):
        self.assertEqual(problems("0x0", "0xc0000005"), [])

    def test_missing_and_killed_exits_keep_distinct_diagnostics(self):
        self.assertTrue(any("no exit code" in p for p in problems("", "0x0")))
        self.assertTrue(any("did not end by itself" in p for p in problems("killed (no exit)", "0x0")))


if __name__ == "__main__":
    unittest.main()
