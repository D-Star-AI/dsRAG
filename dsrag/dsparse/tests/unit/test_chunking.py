import os
import sys
import unittest
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dsparse.models.types import Chunk
from dsparse.sectioning_and_chunking.chunking import find_lines_in_range, chunk_sub_section, chunk_document

class TestChunking(unittest.TestCase):
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

    def test_chunk_document_basic(self):
        """Test basic document chunking without visual elements"""
        document_lines = [
            {"content": f"Line {i}", "is_visual": False, "page_number": 1} 
            for i in range(10)
        ]
        sections = [
            {
                "title": "Section 1",
                "start": 0,
                "end": 4,
                "content": "\n".join([document_lines[i]["content"] for i in range(5)])
            },
            {
                "title": "Section 2", 
                "start": 5,
                "end": 9,
                "content": "\n".join([document_lines[i]["content"] for i in range(5, 10)])
            }
        ]
        
        chunks = chunk_document(
            sections=sections,
            document_lines=document_lines,
            chunk_size=50,
            min_length_for_chunking=100
        )
        
        # Since text is short, should get one chunk per section
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["line_start"], 0)
        self.assertEqual(chunks[0]["line_end"], 4)
        self.assertEqual(chunks[1]["line_start"], 5)
        self.assertEqual(chunks[1]["line_end"], 9)

    def test_chunk_document_with_visual_elements(self):
        """Test chunking with visual elements that should be preserved"""
        document_lines = [
            {"content": "Text line 1", "is_visual": False, "page_number": 1},
            {"content": "Text line 2", "is_visual": False, "page_number": 1},
            {"content": "Image 1", "is_visual": True, "page_number": 1},
            {"content": "Text line 3", "is_visual": False, "page_number": 1},
            {"content": "Text line 4", "is_visual": False, "page_number": 1},
            {"content": "Image 2", "is_visual": True, "page_number": 2},
            {"content": "Text line 5", "is_visual": False, "page_number": 2}
        ]
        sections = [{
            "title": "Section 1",
            "start": 0,
            "end": 6,
            "content": "\n".join([line["content"] for line in document_lines])
        }]
        
        chunks = chunk_document(
            sections=sections,
            document_lines=document_lines,
            chunk_size=50,
            min_length_for_chunking=100
        )
        
        # Should get separate chunks for text and visual elements
        visual_chunks = [c for c in chunks if c["is_visual"]]
        text_chunks = [c for c in chunks if not c["is_visual"]]
        
        self.assertEqual(len(visual_chunks), 2)
        self.assertTrue(all(chunk["is_visual"] for chunk in visual_chunks))
        self.assertTrue(all(not chunk["is_visual"] for chunk in text_chunks))
        
        # Check visual chunks are single lines
        for chunk in visual_chunks:
            self.assertEqual(chunk["line_start"], chunk["line_end"])

    def test_chunk_document_large_text(self):
        """Test chunking of large text sections"""
        # Create a long document that will need multiple chunks
        document_lines = [
            {"content": f"This is line {i} with some additional text to make it longer", "is_visual": False, "page_number": i//10 + 1}
            for i in range(50)
        ]
        sections = [{
            "title": "Long Section",
            "start": 0,
            "end": 49,
            "content": "\n".join([line["content"] for line in document_lines])
        }]
        
        chunks = chunk_document(
            sections=sections,
            document_lines=document_lines,
            chunk_size=200,  # Small chunk size to force multiple chunks
            min_length_for_chunking=100
        )
        
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks
        
        # Verify chunk boundaries
        for i in range(len(chunks) - 1):
            self.assertEqual(chunks[i]["line_end"] + 1, chunks[i + 1]["line_start"],
                            "Chunks should be continuous")
        
        # Verify all chunks reference valid lines
        for chunk in chunks:
            self.assertGreaterEqual(chunk["line_start"], 0)
            self.assertLess(chunk["line_end"], len(document_lines))
            self.assertLess(chunk["line_start"], chunk["line_end"])

    def test_chunk_document_edge_cases(self):
        """Test various edge cases in document chunking"""
        # Test empty section
        document_lines = [{"content": "Line 1", "is_visual": False, "page_number": 1}]
        sections = [{
            "title": "Empty Section",
            "start": 0,
            "end": 0,
            "content": "Line 1"
        }]
        
        chunks = chunk_document(
            sections=sections,
            document_lines=document_lines,
            chunk_size=50,
            min_length_for_chunking=100
        )
        self.assertEqual(len(chunks), 1)
        
        # Test section with only visual elements
        document_lines = [
            {"content": "Image 1", "is_visual": True, "page_number": 1},
            {"content": "Image 2", "is_visual": True, "page_number": 1}
        ]
        sections = [{
            "title": "Visual Section",
            "start": 0,
            "end": 1,
            "content": "Image 1\nImage 2"
        }]
        
        chunks = chunk_document(
            sections=sections,
            document_lines=document_lines,
            chunk_size=50,
            min_length_for_chunking=100
        )
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(chunk["is_visual"] for chunk in chunks))

    def test_find_lines_in_range(self):
        """Test the line range finding function"""
        # Create sample line ranges
        line_char_ranges = [
            (0, 0, 10),   # Line 0: chars 0-10
            (1, 11, 20),  # Line 1: chars 11-20
            (2, 21, 30),  # Line 2: chars 21-30
            (3, 31, 40)   # Line 3: chars 31-40
        ]
        
        # Test exact match
        start, end = find_lines_in_range(0, 10, line_char_ranges, 0, 3)
        self.assertEqual(start, 0)
        self.assertEqual(end, 0)
        
        # Test spanning multiple lines
        start, end = find_lines_in_range(5, 25, line_char_ranges, 0, 3)
        self.assertEqual(start, 0)
        self.assertEqual(end, 2)
        
        # Test partial overlap
        start, end = find_lines_in_range(15, 35, line_char_ranges, 0, 3)
        self.assertEqual(start, 1)
        self.assertEqual(end, 3)

    def test_chunk_sub_section(self):
        """Test chunking of sub-sections"""
        document_lines = [
            {"content": f"Line {i} with some content", "is_visual": False}
            for i in range(10)
        ]
        
        # Test basic chunking
        chunks_text, chunk_indices = chunk_sub_section(0, 9, document_lines, max_length=50)
        self.assertGreater(len(chunks_text), 1)  # Should create multiple chunks
        self.assertEqual(len(chunks_text), len(chunk_indices))
        
        # Test small text that shouldn't be chunked
        chunks_text, chunk_indices = chunk_sub_section(0, 2, document_lines, max_length=1000)
        self.assertEqual(len(chunks_text), 1)  # Should be a single chunk
        
        # Verify chunk boundaries
        for start, end in chunk_indices:
            self.assertGreaterEqual(start, 0)
            self.assertLess(end, len(document_lines))
            self.assertLessEqual(start, end)

if __name__ == "__main__":
    unittest.main()