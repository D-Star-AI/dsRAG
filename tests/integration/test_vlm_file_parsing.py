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
        pass

if __name__ == "__main__":
    unittest.main()