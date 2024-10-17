import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.dsparse.vlm_file_parsing import parse_file
from dsrag.knowledge_base import KnowledgeBase

"""
class TestParsing(unittest.TestCase):
    def test__parse_and_chunk_vlm(self):
        # delete the directory where the images were saved
        try:
            os.system("rm -rf ~/dsrag_test_mck_energy")
            print("Deleted directory ~/dsrag_test_mck_energy")
        except:
            pass
        
        pdf_path = "../data/mck_energy_first_5_pages.pdf"
        save_path = "~/dsrag_test_mck_energy" # base directory to save the page images, pages with bounding boxes, and extracted images

        # convert pdf path to absolute path because pdf2image requires it
        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), pdf_path))

        vlm_config = {
            "provider": "vertex_ai",
            "model": "gemini-1.5-pro-002",
            "project_id": os.environ["VERTEX_PROJECT_ID"],
            "location": "us-central1"
        }
        
        all_page_content = parse_file(pdf_path, save_path, vlm_config)

        # assert that it found the correct number of images
        num_images = len([content for content in all_page_content if content["type"] == "Image"])
        self.assertEqual(num_images, 2)

        # delete the directory where the images were saved
        os.system("rm -rf ~/dsrag_test_mck_energy")
"""

class TestRetrieval(unittest.TestCase):
    def test__retrieval(self):
        self.cleanup()  # delete the KnowledgeBase object if it exists so we can start fresh

        file_path = "../data/mck_energy_first_5_pages.pdf"

        # convert file path to absolute path because pdf2image requires it
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
        print(file_path)

        kb = KnowledgeBase(kb_id="mck_energy")
        kb.add_document(
            doc_id="mck_energy_report",
            file_path=file_path,
            document_title="McKinsey Energy Report",
            file_parsing_config={
                "use_vlm": True,
                "vlm_config": {
                    "provider": "vertex_ai",
                    "model": "gemini-1.5-pro-002",
                    "project_id": os.environ["VERTEX_PROJECT_ID"],
                    "location": "us-central1",
                    "save_path": "~/dsrag_test_mck_energy"
                }
            }
        )

        kb = KnowledgeBase(kb_id="mck_energy")
        
        query = "People stacking boxes"
        rse_params = {
            "minimum_value": 0.0,
            "irrelevant_chunk_penalty": 0.1,
        }

        search_results = kb.query(search_queries=[query], rse_params=rse_params, return_images=True)
        print (search_results)

        first_result = search_results[0]
        self.assertEqual(first_result["segment_page_start"], 3)
        self.assertEqual(first_result["segment_page_end"], 3)

        first_result_content = first_result["content"]
        is_image = [content for content in first_result_content if content["type"] == "image"]
        self.assertEqual(len(is_image), 1)

        self.cleanup()

    def cleanup(self):
        kb = KnowledgeBase(kb_id="mck_energy", exists_ok=True)
        kb.delete()

        # delete the directory where the images were saved
        os.system("rm -rf ~/dsrag_test_mck_energy")

        
if __name__ == "__main__":
    unittest.main()