import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.dsparse.file_parsing.file_system import LocalFileSystem
from dsrag.knowledge_base import KnowledgeBase


class TestRetrieval(unittest.TestCase):
    def test__retrieval(self):
        self.cleanup()  # delete the KnowledgeBase object if it exists so we can start fresh

        file_path = "../data/mck_energy_first_5_pages.pdf"

        # convert file path to absolute path because pdf2image requires it
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
        print(file_path)

        save_path = "~/dsrag_test_mck_energy"

        file_system = LocalFileSystem(base_path=save_path)
        vlm_config = {
            "provider": "gemini",
            "model": "gemini-1.5-flash-002",
        }
        file_parsing_config = {
            "use_vlm": True,
            "vlm_config": vlm_config
        }

        kb = KnowledgeBase(kb_id="mck_energy_test", file_system=file_system)
        kb.add_document(
            doc_id="mck_energy_report",
            file_path=file_path,
            document_title="McKinsey Energy Report",
            file_parsing_config=file_parsing_config
        )

        #kb = KnowledgeBase(kb_id="mck_energy_test")
        
        query = "Image of people collaborating around a structure, with a car in the background and people flying kites in the sky"
        rse_params = {
            "minimum_value": 0.0,
            "irrelevant_chunk_penalty": 0.5,
        }

        search_results = kb.query(search_queries=[query], rse_params=rse_params, return_mode="page_images")

        first_result = search_results[0]
        self.assertTrue(first_result["segment_page_start"] == 1 or first_result["segment_page_start"] == 3)
        self.assertTrue(first_result["segment_page_end"] == 1 or first_result["segment_page_end"] == 3)
        first_result_content = first_result["content"]
        # Make sure the first result is a png image
        self.assertTrue(first_result_content[0].endswith(".png"))


        search_results = kb.query(search_queries=[query], rse_params=rse_params, return_mode="text")
        first_result = search_results[0]
        first_result_content = first_result["content"]
        self.assertTrue(type(first_result_content) == str)

        # Test dynamic mode. This should return the same result as page_image mode
        search_results = kb.query(search_queries=[query], rse_params=rse_params, return_mode="dynamic")
        first_result = search_results[0]
        self.assertTrue(first_result["segment_page_start"] == 1 or first_result["segment_page_start"] == 3)
        self.assertTrue(first_result["segment_page_end"] == 1 or first_result["segment_page_end"] == 3)
        first_result_content = first_result["content"]
        # Make sure the first result is a png image
        self.assertTrue(first_result_content[0].endswith(".png"))


        self.cleanup()


    def test__retrieval_non_vlm(self):
        self.cleanup()  # delete the KnowledgeBase object if it exists so we can start fresh

        file_path = "../data/mck_energy_first_5_pages.pdf"

        # convert file path to absolute path because pdf2image requires it
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
        print(file_path)

        save_path = "~/dsrag_test_mck_energy"

        file_system = LocalFileSystem(base_path=save_path)
        file_parsing_config = {
            "use_vlm": False,
            "always_save_page_images": True
        }

        kb = KnowledgeBase(kb_id="mck_energy_test", file_system=file_system)
        kb.add_document(
            doc_id="mck_energy_report",
            file_path=file_path,
            document_title="McKinsey Energy Report",
            file_parsing_config=file_parsing_config
        )
        
        query = "Should new energy businesses be integrated in or independent from the core?"
        rse_params = {
            "minimum_value": 0.0,
            "irrelevant_chunk_penalty": 0.25,
        }

        search_results = kb.query(search_queries=[query], rse_params=rse_params, return_mode="page_images")
        print ("search_results: ", search_results)
        print ("\n")
        print ("---------------------------------")
        print ("\n")
        first_result = search_results[0]
        self.assertTrue(first_result["segment_page_start"] == 5)
        self.assertTrue(first_result["segment_page_end"] == 5)
        first_result_content = first_result["content"]
        # Make sure the first result is a png image
        self.assertTrue(first_result_content[0].endswith(".png"))


        search_results = kb.query(search_queries=[query], rse_params=rse_params, return_mode="text")
        first_result = search_results[0]
        first_result_content = first_result["content"]
        self.assertTrue(type(first_result_content) == str)

        # Test dynamic mode. This should return the same result as page_image mode
        search_results = kb.query(search_queries=[query], rse_params=rse_params, return_mode="dynamic")
        print ("search_results dynamic: ", search_results)
        first_result = search_results[0]
        self.assertTrue(first_result["segment_page_start"] == 5)
        self.assertTrue(first_result["segment_page_end"] == 5)
        first_result_content = first_result["content"]
        # Make sure the first result is text content
        self.assertTrue(type(first_result_content) == str)


        self.cleanup()

    def test_page_content_storage(self):
        """Test that page content is properly stored and retrieved when using VLM."""
        self.cleanup()

        file_path = "../data/mck_energy_first_5_pages.pdf"
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
        save_path = "~/dsrag_test_mck_energy"
        save_path = os.path.expanduser(save_path)  # Expand the ~ to full path

        file_system = LocalFileSystem(base_path=save_path)
        vlm_config = {
            "provider": "gemini",
            "model": "gemini-1.5-flash-002",
        }
        file_parsing_config = {
            "use_vlm": True,
            "vlm_config": vlm_config
        }

        # Create KB and add document
        kb = KnowledgeBase(kb_id="mck_energy_test", file_system=file_system)
        kb.add_document(
            doc_id="mck_energy_report",
            file_path=file_path,
            document_title="McKinsey Energy Report",
            file_parsing_config=file_parsing_config
        )

        # Verify the page content files exist
        doc_dir = os.path.join(save_path, "mck_energy_test", "mck_energy_report")
        self.assertTrue(os.path.exists(doc_dir))
        
        # Check that page content files were created
        page_content_files = [f for f in os.listdir(doc_dir) if f.startswith('page_content_') and f.endswith('.json')]
        self.assertTrue(len(page_content_files) > 0)
        
        # Test loading content for a specific page
        page_content = kb.file_system.load_page_content(
            kb_id="mck_energy_test",
            doc_id="mck_energy_report",
            page_number=1
        )
        self.assertIsNotNone(page_content)
        self.assertTrue(isinstance(page_content, str))
        self.assertTrue(len(page_content) > 0)

        # Test loading a range of pages
        page_contents = kb.file_system.load_page_content_range(
            kb_id="mck_energy_test",
            doc_id="mck_energy_report",
            page_start=1,
            page_end=3
        )
        self.assertEqual(len(page_contents), 3)
        for content in page_contents:
            self.assertTrue(isinstance(content, str))
            self.assertTrue(len(content) > 0)

        # Verify the content format
        import json
        page_content_path = os.path.join(doc_dir, 'page_content_1.json')
        with open(page_content_path, 'r') as f:
            content_data = json.load(f)
            self.assertIn('content', content_data)
            self.assertTrue(isinstance(content_data['content'], str))

        # Test that non-VLM document doesn't create page content
        kb.add_document(
            doc_id="text_doc",
            text="This is a test document without page numbers.",
            document_title="Test Document"
        )

        # Verify no page content was created for the text document
        text_doc_dir = os.path.join(save_path, "mck_energy_test", "text_doc")
        if os.path.exists(text_doc_dir):
            page_content_files = [f for f in os.listdir(text_doc_dir) if f.startswith('page_content_')]
            self.assertEqual(len(page_content_files), 0)

        text_doc_content = kb.file_system.load_page_content(
            kb_id="mck_energy_test",
            doc_id="text_doc",
            page_number=1
        )
        self.assertIsNone(text_doc_content)

        self.cleanup()

    def cleanup(self):
        kb = KnowledgeBase(kb_id="mck_energy_test", exists_ok=True)
        kb.delete()

        # delete the directory where the images were saved
        os.system("rm -rf ~/dsrag_test_mck_energy")

        
if __name__ == "__main__":
    unittest.main()