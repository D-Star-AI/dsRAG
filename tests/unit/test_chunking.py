import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.dsparse.sectioning_and_chunking.chunking import chunk_document, chunk_sub_section, find_lines_in_range

class TestChunking(unittest.TestCase):
    def setUp(self):
        self.sections = [
            {
                'title': 'Section 1',
                'start': 0,
                'end': 3,
                'content': 'This is the first section of the document.'
            },
            {
                'title': 'Section 2',
                'start': 4,
                'end': 7,
                'content': 'This is the second section of the document.'
            }
        ]

        self.document_lines = [
            {
                'content': 'This is the first line of the document. And here is another sentence.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'image_path': None
            },
            {
                'content': 'This is the second line of the document.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'image_path': None
            },
            {
                'content': 'This is the third line of the document........',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'image_path': None
            },
            {
                'content': 'This is the fourth line of the document.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'image_path': None
            },
            {
                'content': 'This is the fifth line of the document. With another sentence.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'image_path': None
            },
            {
                'content': 'This is the sixth line of the document.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'image_path': None
            },
            {
                'content': 'This is the seventh line of the document.',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'image_path': None
            },
            {
                'content': 'This is the eighth line of the document. And here is another sentence that is a bit longer',
                'element_type': 'NarrativeText',
                'page_number': 1,
                'image_path': None
            }
        ]

    def test__chunk_document(self):
        chunk_size = 90
        min_length_for_chunking = 120
        chunks = chunk_document(self.sections, self.document_lines, chunk_size, min_length_for_chunking)
        assert len(chunks[-1]["content"]) > 45

    def test__chunk_sub_section(self):
        chunk_size = 90
        chunks_text, chunk_line_indices = chunk_sub_section(4, 7, self.document_lines, chunk_size)
        assert len(chunks_text) == 3
        assert chunk_line_indices == [(4, 4), (5, 6), (7, 7)]

    def test__find_lines_in_range(self):
        # Test find_lines_in_range
        # (line_idx, start_char, end_char)
        line_char_ranges = [
            (0, 0, 49),
            (1, 50, 99),
            (2, 100, 149),
            (3, 150, 199),
            (4, 200, 249),
            (5, 250, 299),
            (6, 300, 349),
            (7, 350, 399)
        ]

        chunk_start = 50
        chunk_end = 82
        line_start = 0
        line_end = 0
        chunk_line_start, chunk_line_end = find_lines_in_range(chunk_start, chunk_end, line_char_ranges, line_start, line_end)
        assert chunk_line_start == 1
        assert chunk_line_end == 1

        chunk_start = 50
        chunk_end = 150
        line_start = 0
        line_end = 0
        chunk_line_start, chunk_line_end = find_lines_in_range(chunk_start, chunk_end, line_char_ranges, line_start, line_end)
        assert chunk_line_start == 1
        assert chunk_line_end == 3

if __name__ == "__main__":
    unittest.main()