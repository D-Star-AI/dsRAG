import os
import sys
import unittest
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dsparse.sectioning_and_chunking.chunking import find_lines_in_range, chunk_sub_section, chunk_document
from dsparse.models.types import Chunk
from dsparse.sectioning_and_chunking.semantic_sectioning import get_document_with_lines

class TestDsParse(unittest.TestCase):

    def test__chunk_document(self):

        # Load in the test data
        document_lines_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/document_lines.json'))
        with open(document_lines_path, 'r') as f:
            document_lines = json.load(f)
        sections_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/sections.json'))
        with open(sections_path, 'r') as f:
            sections = json.load(f)

        chunks = chunk_document(
            sections=sections, 
            document_lines=document_lines, 
            chunk_size=800, 
            min_length_for_chunking=1600
        )

        # Get the keys and types from the Chunks TypedDict
        chunk_keys_and_types = Chunk.__annotations__.items()

        # Check if the first element in chunks is a dictionary with the expected keys
        self.assertIsInstance(chunks[0], dict)
        for key, _ in chunk_keys_and_types:
            self.assertIn(key, chunks[0])

        self.assertEqual(chunks[0]['line_start'], 0)
        self.assertEqual(chunks[0]['line_end'], 5)
        self.assertEqual(chunks[0]['is_visual'], False)
        self.assertEqual(chunks[0]['page_start'], 1)
        self.assertEqual(chunks[0]['page_end'], 1)
        self.assertEqual(chunks[0]['section_index'], 0)

    def test__get_document_with_lines(self):

        document_lines_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/document_lines.json'))
        with open(document_lines_path, 'r') as f:
            document_lines = json.load(f)

        print (len(document_lines))

        document_with_line_numbers, end_line = get_document_with_lines(document_lines, start_line=0, max_characters=20000)
        split_lines = document_with_line_numbers.split("\n")

        self.assertEqual(split_lines[0], "[0] McKinsey")
        self.assertEqual(end_line, 203)
    
    def test__find_lines_in_range(self):
        pass

    def test__chunk_sub_section(self):
        pass


if __name__ == "__main__":
    unittest.main()