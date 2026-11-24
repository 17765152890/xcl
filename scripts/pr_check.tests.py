"""Tests of scripts/pr_check.py's check of the description's sections. No GitHub, no git:

    python3 scripts/pr_check.tests.py
"""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pr_check

HEAD = "0123456789abcdef0123456789abcdef01234567"
FILLED = """A change to a script.

**What is in it**
- `scripts/example.py`: the change

**Tested**
- `python3 scripts/example.py`: ok

**Not tested**
- Windows

**AI use**
- none
"""


def problems(body):
    # one file that is not runtime: only the description is checked
    files = [("scripts/example.py", "modified", False, lambda: 0)]
    return pr_check.check(body, HEAD, files, lambda sha: False)


def empty(section):
    return f"the **{section}** section is missing or empty"


class Sections(unittest.TestCase):
    def test_filled_description_passes(self):
        self.assertEqual(problems(FILLED), [])
        self.assertEqual(problems(FILLED.replace("\n", "\r\n")), [])

    def test_missing_section(self):
        body = FILLED.replace("**Not tested**\n- Windows\n\n", "")
        self.assertEqual(problems(body), [empty("Not tested")])

    def test_empty_section_before_another_one(self):
        # the next section's heading is not this section's text
        body = FILLED.replace("- `python3 scripts/example.py`: ok\n", "")
        self.assertEqual(problems(body), [empty("Tested")])
        body = FILLED.replace("- Windows\n", "")
        self.assertEqual(problems(body), [empty("Not tested")])

    def test_empty_last_section(self):
        self.assertEqual(problems(FILLED.replace("- none\n", "")), [empty("AI use")])

    def test_list_marker_alone_is_empty(self):
        body = FILLED.replace("- Windows\n", "-\n")
        self.assertEqual(problems(body), [empty("Not tested")])
        body = FILLED.replace("- Windows\n", "- \n  *\n")
        self.assertEqual(problems(body), [empty("Not tested")])

    def test_comment_alone_is_empty(self):
        body = FILLED.replace("- Windows\n", "<!-- what you could\n     not check -->\n")
        self.assertEqual(problems(body), [empty("Not tested")])

    def test_section_inside_a_comment_is_not_there(self):
        # the heading and its only result, both commented out
        body = FILLED.replace("**Tested**\n- `python3 scripts/example.py`: ok\n",
                              "<!--\n**Tested**\n- `python3 scripts/example.py`: ok\n-->\n")
        self.assertEqual(problems(body), [empty("Tested")])
        # the heading is there, its only result is commented out across lines
        body = FILLED.replace("- Windows\n", "<!--\n- Windows\n-->\n")
        self.assertEqual(problems(body), [empty("Not tested")])

    def test_heading_is_a_line_of_its_own(self):
        # a section name in bold at the end of a sentence is not the section's heading
        body = FILLED.replace("**Tested**\n- `python3 scripts/example.py`: ok\n",
                              "Nothing here was **Tested**\nas far as I know.\n")
        self.assertEqual(problems(body), [empty("Tested")])
        # indented, or with spaces after it, it still is
        body = FILLED.replace("**Tested**\n", "  **Tested**  \n")
        self.assertEqual(problems(body), [])

    def test_untouched_template_is_refused(self):
        path = os.path.join(HERE, "..", ".github", "pull_request_template.md")
        with open(path, encoding="utf-8") as f:
            self.assertIn(empty("Not tested"), problems(f.read()))

    def test_text_in_bold_is_text(self):
        # a line in bold that is not a heading of the outline does not end the section
        body = FILLED.replace("- Windows\n", "**AMD**: not run\n")
        self.assertEqual(problems(body), [])
        body = FILLED.replace("- Windows\n", "\n\n  not on AMD\n")
        self.assertEqual(problems(body), [])


if __name__ == "__main__":
    unittest.main()
