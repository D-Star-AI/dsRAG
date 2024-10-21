import os
import sys
import unittest
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dsrag.dsparse.parse_and_chunk import parse_and_chunk_vlm
from dsrag.dsparse.types import Chunks, Sections


class TestDsParse(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dsparse_output'))
        try:
            shutil.rmtree(self.save_path)
        except:
            pass

    def test__parse_and_chunk_vlm(self):

        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/mck_energy_first_5_pages.pdf'))
        vlm_config = {
            "provider": "gemini",
            "model": "gemini-1.5-pro-002",
            "save_path": self.save_path
        }
        semantic_sectioning_config = {
            "llm_provider": "openai",
            "model": "gpt-4o-mini",
            "language": "en",
        }
        sections, chunks = parse_and_chunk_vlm(
            file_path=file_path,
            vlm_config=vlm_config,
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config={}
        )

        # Make sure the sections and chunks were created, and are the correct types
        self.assertTrue(len(sections) > 0)
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(type(sections), list)
        self.assertEqual(type(chunks), list)
        
        for key, expected_type in Sections.__annotations__.items():
            self.assertIsInstance(sections[0][key], expected_type)
        
        for key, expected_type in Chunks.__annotations__.items():
            self.assertIsInstance(chunks[0][key], expected_type)

        self.assertTrue(len(sections[0]["title"]) > 0)
        self.assertTrue(len(sections[0]["content"]) > 0)

        # Make sure the elements.json file was created
        self.assertTrue(os.path.exists(os.path.join(self.save_path, "elements.json")))

        # Make sure the extracted_images were created
        self.assertTrue(os.path.exists(os.path.join(self.save_path, "extracted_images")))

    @classmethod
    def tearDownClass(self):
        # Delete the save path
        shutil.rmtree(self.save_path)

if __name__ == "__main__":
    unittest.main()