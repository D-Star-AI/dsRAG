import os
import sys
import shutil
import tempfile
import unittest

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from dsrag.dsparse.file_parsing.vlm_file_parsing import parse_file
from dsrag.dsparse.file_parsing.file_system import LocalFileSystem


class TestVLMConfigTypedDictAlignment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="dsparse_vlm_cfg_")
        self.kb_id = "kb_vlm_cfg"
        self.doc_id = "doc_vlm_cfg"
        self.fs = LocalFileSystem(base_path=self.temp_dir)
        # Ensure directory exists but contains no images
        self.fs.create_directory(self.kb_id, self.doc_id)

    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_vlm_config_typedict_alignment(self):
        # Provide dpi and vlm_max_concurrent_requests; zero images and images_already_exist=True
        vlm_config = {
            "images_already_exist": True,
            "dpi": 120,
            "vlm_max_concurrent_requests": 2,
        }
        # pdf_path can be empty when images_already_exist is True
        elements = parse_file(
            pdf_path="",
            kb_id=self.kb_id,
            doc_id=self.doc_id,
            vlm_config=vlm_config,  # type: ignore[arg-type]
            file_system=self.fs,
        )
        # Should return an empty list and write elements.json
        self.assertIsInstance(elements, list)
        self.assertEqual(len(elements), 0)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, self.kb_id, self.doc_id, "elements.json")))


if __name__ == "__main__":
    unittest.main()
