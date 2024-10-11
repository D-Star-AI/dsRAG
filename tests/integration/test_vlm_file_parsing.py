import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.dsparse.parse_and_chunk import parse_file
from dsrag.knowledge_base import KnowledgeBase


class TestParsing(unittest.TestCase):
    def test__parse_and_chunk_vlm(self):
        user_id = "zmcc" 
        pdf_path = "/Users/zach/Code/mck_energy_first_5_pages.pdf"
        file_id = "mck_energy"

        save_path = f"{user_id}/{file_id}" # base directory to save the page images, pages with bounding boxes, and extracted images

        vlm_config = {
            "provider": "vertex_ai",
            "model": "gemini-1.5-pro-002",
            "project_id": os.environ["VERTEX_PROJECT_ID"],
            "location": "us-central1"
        }
        
        all_page_content = parse_file(pdf_path, save_path, vlm_config)

        # assert that it found the correct number of images
        num_images = len([content for content in all_page_content if content["type"] == "image"])
        self.assertEqual(num_images, 2)

class TestRetrieval(unittest.TestCase):
    def test__retrieval(self):
        self.cleanup()  # delete the KnowledgeBase object if it exists so we can start fresh

        file_path = "/Users/zach/Code/mck_energy_first_5_pages.pdf"
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
                    "save_path": "zmcc/mck_energy"
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
        self.assertEqual(first_result["chunk_page_start"], 3)
        self.assertEqual(first_result["chunk_page_end"], 3)

        first_result_content = first_result["content"]
        is_image = [content for content in first_result_content if content["type"] == "image"]
        self.assertEqual(len(is_image), 1)

    def cleanup(self):
        kb = KnowledgeBase(kb_id="mck_energy", exists_ok=True)
        kb.delete()

if __name__ == "__main__":
    unittest.main()