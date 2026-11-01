import json
import struct
import unittest

from trellis2_smoke import glb_document


class GlbDocumentTests(unittest.TestCase):
    def test_reads_a_version_two_glb(self):
        document = {"asset": {"version": "2.0"}, "materials": [{}]}
        chunk = json.dumps(document).encode()
        chunk += b" " * (-len(chunk) % 4)
        data = struct.pack("<4sIIII", b"glTF", 2, 20 + len(chunk), len(chunk), 0x4E4F534A)
        self.assertEqual(glb_document(data + chunk), document)

    def test_rejects_short_output(self):
        with self.assertRaises(ValueError):
            glb_document(b"not a GLB")

    def test_rejects_incorrect_length_or_version(self):
        for version, length in ((1, 20), (2, 100)):
            with self.subTest(version=version, length=length):
                data = struct.pack("<4sIIII", b"glTF", version, length, 0, 0x4E4F534A)
                with self.assertRaises(ValueError):
                    glb_document(data)

    def test_rejects_non_json_first_chunk(self):
        data = struct.pack("<4sIIII", b"glTF", 2, 20, 0, 0x004E4942)
        with self.assertRaises(ValueError):
            glb_document(data)


if __name__ == "__main__":
    unittest.main()
