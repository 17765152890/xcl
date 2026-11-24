from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RecordWritersTests(unittest.TestCase):
    def test_branch_target_accepts_both_hex_spellings(self):
        script = Path(__file__).parent / "re" / "record_writers.py"
        listing = """\
  140001000: mov 0x18(%rcx),%rax
  140001004: cmp 0x30(%rcx),%rax
  140001008: jae {target} <fallback>
  14000100c: movl $0x37,(%rax)
  140001010: add $0x8,%rax
  140001014: mov %rax,0x18(%rcx)
  140001020: call *0x48(%rdx)
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "writer.asm"
            for target in ("140001020", "0x140001020"):
                with self.subTest(target=target):
                    path.write_text(listing.format(target=target), encoding="ascii")
                    output = subprocess.run([sys.executable, str(script), str(path)], check=True,
                                            capture_output=True, text=True).stdout
                    self.assertIn("1 record write sites", output)
                    self.assertIn("fallback slot 9", output)
