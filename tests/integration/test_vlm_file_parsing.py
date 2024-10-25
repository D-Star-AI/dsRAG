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

    def cleanup(self):
        kb = KnowledgeBase(kb_id="mck_energy_test", exists_ok=True)
        kb.delete()

        # delete the directory where the images were saved
        os.system("rm -rf ~/dsrag_test_mck_energy")

        
if __name__ == "__main__":
    unittest.main()