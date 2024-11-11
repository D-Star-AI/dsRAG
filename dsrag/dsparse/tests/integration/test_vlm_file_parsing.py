import os
import sys
import unittest
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dsparse.main import parse_and_chunk
from dsparse.models.types import Chunk, Section
from dsparse.file_parsing.file_system import LocalFileSystem


class TestVLMFileParsing(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dsparse_output'))
        self.file_system = LocalFileSystem(base_path=self.save_path)
        self.kb_id = "test_kb"
        self.doc_id = "test_doc"
        self.test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../tests/data/mck_energy_first_5_pages.pdf'))
        try:
            shutil.rmtree(self.save_path)
        except:
            pass

    def test__parse_and_chunk_vlm(self):

        vlm_config = {
            "provider": "gemini",
            "model": "gemini-1.5-flash-002",
        }
        semantic_sectioning_config = {
            "llm_provider": "openai",
            "model": "gpt-4o-mini",
            "language": "en",
        }
        file_parsing_config = {
            "use_vlm": True,
            "vlm_config": vlm_config,
            "always_save_page_images": True
        }
        sections, chunks = parse_and_chunk(
            kb_id=self.kb_id,
            doc_id=self.doc_id,
            file_path=self.test_data_path,
            file_parsing_config=file_parsing_config,
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config={},
            file_system=self.file_system,
        )

        # Make sure the sections and chunks were created, and are the correct types
        self.assertTrue(len(sections) > 0)
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(type(sections), list)
        self.assertEqual(type(chunks), list)
        
        for key, expected_type in Section.__annotations__.items():
            self.assertIsInstance(sections[0][key], expected_type)
        
        for key, expected_type in Chunk.__annotations__.items():
            self.assertIsInstance(chunks[0][key], expected_type)

        self.assertTrue(len(sections[0]["title"]) > 0)
        self.assertTrue(len(sections[0]["content"]) > 0)

        # Make sure the elements.json file was created
        self.assertTrue(os.path.exists(os.path.join(self.save_path, self.kb_id, self.doc_id, "elements.json")))

        # Delete the save path
        try:
            shutil.rmtree(self.save_path)
        except:
            pass


    def test_non_vlm_file_parsing(self):

        semantic_sectioning_config = {
            "llm_provider": "openai",
            "model": "gpt-4o-mini",
            "language": "en",
        }
        file_parsing_config = {
            "use_vlm": False,
            "always_save_page_images": True,
        }
        sections, chunks = parse_and_chunk(
            kb_id=self.kb_id,
            doc_id=self.doc_id,
            file_path=self.test_data_path,
            file_parsing_config=file_parsing_config,
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config={},
            file_system=self.file_system,
        )
        
        # Make sure the sections and chunks were created, and are the correct types
        self.assertTrue(len(sections) > 0)
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(type(sections), list)
        self.assertEqual(type(chunks), list)
        
        for key, expected_type in Section.__annotations__.items():
            self.assertIsInstance(sections[0][key], expected_type)
        
        for key, expected_type in Chunk.__annotations__.items():
            self.assertIsInstance(chunks[0][key], expected_type)

        self.assertTrue(len(sections[0]["title"]) > 0)
        self.assertTrue(len(sections[0]["content"]) > 0)


    @classmethod
    def tearDownClass(self):
        # Delete the save path
        try:
            shutil.rmtree(self.save_path)
        except:
            pass

if __name__ == "__main__":
    unittest.main()