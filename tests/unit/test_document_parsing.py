import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.document_parsing import get_pages_from_chunks

class TestPDFPages(unittest.TestCase):

    def test_get_pdf_pages(self):

        full_text = "This is some sample text for testing purposes. It doesn't really matter what is in here, just something to test with."
        pages = [
            "This is some sample text for testing purposes.",
            "It doesn't really matter what is in here, just something to test with."
        ]
        chunks = [
            {"chunk_text": "This is some sample text for"},
            {"chunk_text": "testing purposes. It doesn't really matter"},
            {"chunk_text": "what is in here, just something to test with."}
        ]

        formatted_chunks = get_pages_from_chunks(full_text, pages, chunks)

        self.assertEqual(len(formatted_chunks), 3)
        self.assertEqual(formatted_chunks[0]["chunk_page_start"], 1)
        self.assertEqual(formatted_chunks[0]["chunk_page_end"], 1)
        self.assertEqual(formatted_chunks[1]["chunk_page_start"], 1)
        self.assertEqual(formatted_chunks[1]["chunk_page_end"], 2)
        self.assertEqual(formatted_chunks[2]["chunk_page_start"], 2)
        self.assertEqual(formatted_chunks[2]["chunk_page_end"], 2)


if __name__ == "__main__":
    unittest.main()
