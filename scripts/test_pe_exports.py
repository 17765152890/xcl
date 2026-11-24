from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


class PeExportsTests(unittest.TestCase):
    def test_aliases_share_an_ordinal_row(self):
        script = Path(__file__).parent / "re" / "pe_exports.py"
        pe = bytearray(0x400)
        pe[:2] = b"MZ"
        struct.pack_into("<I", pe, 0x3C, 0x80)
        pe[0x80:0x84] = b"PE\0\0"
        struct.pack_into("<H", pe, 0x86, 1)
        struct.pack_into("<H", pe, 0x94, 0xF0)
        struct.pack_into("<H", pe, 0x98, 0x20B)
        struct.pack_into("<2I", pe, 0x98 + 112, 0x1000, 0x80)
        struct.pack_into("<4I", pe, 0x188 + 8, 0x200, 0x1000, 0x200, 0x200)

        # Export directory, function RVAs, name RVAs and name ordinals.
        struct.pack_into("<7I", pe, 0x20C, 0x1050, 1, 2, 2, 0x1028, 0x1030, 0x1038)
        struct.pack_into("<2I", pe, 0x228, 0x1100, 0x1110)
        struct.pack_into("<2I", pe, 0x230, 0x1060, 0x1068)
        struct.pack_into("<2H", pe, 0x238, 0, 0)
        pe[0x250:0x25A] = b"alias.dll\0"
        pe[0x260:0x266] = b"Alpha\0"
        pe[0x268:0x26D] = b"Beta\0"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.dll"
            path.write_bytes(pe)
            output = subprocess.run([sys.executable, str(script), str(path)], check=True,
                                    capture_output=True, text=True).stdout
        self.assertIn("Alpha, Beta @0x1100", output)
        self.assertIn("(ordinal only) @0x1110", output)
