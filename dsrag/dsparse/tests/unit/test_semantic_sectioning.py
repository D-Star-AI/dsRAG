import os
import sys
import unittest
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dsparse.sectioning_and_chunking.semantic_sectioning import (
    get_document_with_lines, 
    validate_and_fix_sections, 
    split_long_line, 
    elements_to_lines, 
    str_to_lines, 
    pages_to_lines, 
    DocumentSection, 
    get_sections_text,
)

class TestSemanticSectioning(unittest.TestCase):
    def test__get_document_with_lines(self):
        # Test that the document with line numbers is returned correctly
        document_lines_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/document_lines.json'))
        with open(document_lines_path, 'r') as f:
            document_lines = json.load(f)

        print (len(document_lines))

        document_with_line_numbers, end_line = get_document_with_lines(document_lines, start_line=0, max_characters=20000)
        split_lines = document_with_line_numbers.split("\n")

        self.assertEqual(split_lines[0], "[0] McKinsey")
        self.assertEqual(end_line, 203)
    
    def test_validate_and_fix_sections(self):
        # Test with valid sections
        sections = [
            DocumentSection(title="Section 1", start_index=0),
            DocumentSection(title="Section 2", start_index=5),
            DocumentSection(title="Section 3", start_index=10)
        ]
        fixed = validate_and_fix_sections(sections, document_length=15)
        self.assertEqual(len(fixed), 3)
        self.assertEqual([s.start_index for s in fixed], [0, 5, 10])

        # Test with out of order sections
        sections = [
            DocumentSection(title="Section 2", start_index=5),
            DocumentSection(title="Section 1", start_index=0),
            DocumentSection(title="Section 3", start_index=10)
        ]
        fixed = validate_and_fix_sections(sections, document_length=15)
        self.assertEqual([s.start_index for s in fixed], [0, 5, 10])
        self.assertEqual(fixed[0].title, "Section 1")
        self.assertEqual(fixed[1].title, "Section 2")
        self.assertEqual(fixed[2].title, "Section 3")

        # Test with duplicate start indices
        sections = [
            DocumentSection(title="Section 1", start_index=0),
            DocumentSection(title="Section 2", start_index=5),
            DocumentSection(title="Section 3", start_index=5)
        ]
        fixed = validate_and_fix_sections(sections, document_length=15)
        self.assertEqual(len(fixed), 2)
        self.assertEqual([s.start_index for s in fixed], [0, 5])
        self.assertEqual(fixed[0].title, "Section 1")
        self.assertEqual(fixed[1].title, "Section 2")

    def test_split_long_line(self):
        # Test line that doesn't need splitting
        short_line = "This is a short line"
        self.assertEqual(split_long_line(short_line, max_line_length=50), [short_line])

        # Test line that needs splitting
        long_line = "This is a very long line that needs to be split into multiple lines because it exceeds the maximum length"
        split_lines = split_long_line(long_line, max_line_length=20)
        self.assertTrue(all(len(line) <= 20 for line in split_lines))
        self.assertEqual(" ".join(split_lines), long_line)

    def test_elements_to_lines(self):
        elements = [
            {"type": "Title", "content": "Test Document"},
            {"type": "NarrativeText", "content": "This is a test.\nThis is another line."},
            {"type": "Table", "content": "Table content"},
            {"type": "Image", "content": "Image description"}
        ]
        
        # Test with no exclusions
        lines = elements_to_lines(elements, exclude_elements=[], visual_elements=["Image"])
        self.assertTrue(any(line["is_visual"] for line in lines))
        self.assertTrue(any(not line["is_visual"] for line in lines))

        # Test with exclusions
        lines = elements_to_lines(elements, exclude_elements=["Table"], visual_elements=["Image"])
        self.assertFalse(any(line["element_type"] == "Table" for line in lines))

    def test_str_to_lines(self):
        text = "Line 1\nLine 2\nA very long line that should be split into multiple lines"
        lines = str_to_lines(text, max_line_length=20)
        self.assertTrue(all(len(line["content"]) <= 20 for line in lines))
        self.assertTrue(all(line["element_type"] == "NarrativeText" for line in lines))

    def test_pages_to_lines(self):
        pages = [
            "Page 1 content\nMore content",
            "Page 2 content\nEven more content"
        ]
        lines = pages_to_lines(pages)
        self.assertEqual(lines[0]["page_number"], 1)
        self.assertEqual(lines[2]["page_number"], 2)

    def test_validate_and_fix_sections_bounds(self):
        # Test with section starting beyond document length
        sections = [
            DocumentSection(title="Section 1", start_index=0),
            DocumentSection(title="Section 2", start_index=20)  # Beyond doc length
        ]
        fixed = validate_and_fix_sections(sections, document_length=15)
        self.assertEqual(len(fixed), 1)
        self.assertEqual(fixed[0].start_index, 0)

        # Test with all sections beyond document length
        sections = [
            DocumentSection(title="Section 1", start_index=20),
            DocumentSection(title="Section 2", start_index=25)
        ]
        fixed = validate_and_fix_sections(sections, document_length=15)
        self.assertEqual(len(fixed), 1)
        self.assertEqual(fixed[0].start_index, 0)

        # Test with section that are out of order with one section beyond document length
        sections = [
            DocumentSection(title="Section 1", start_index=0),
            DocumentSection(title="Section 3", start_index=20),
            DocumentSection(title="Section 2", start_index=10)
        ]
        fixed = validate_and_fix_sections(sections, document_length=15)
        self.assertEqual(len(fixed), 2)
        self.assertEqual(fixed[0].start_index, 0)
        self.assertEqual(fixed[1].start_index, 10)
        self.assertEqual(fixed[0].title, "Section 1")
        self.assertEqual(fixed[1].title, "Section 2")

    def test_get_sections_text_bounds(self):
        document_lines = [
            {"content": f"Line {i}"} for i in range(10)
        ]
        
        # Test with section ending beyond document length
        sections = [
            DocumentSection(title="Section 1", start_index=0),
            DocumentSection(title="Section 2", start_index=8)
        ]
        result = get_sections_text(sections, document_lines)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1]["end"], 9)

    def test_get_sections_iterative_processing(self):
        """Test the iterative document processing to ensure sections align properly"""
        # Create a document that needs multiple iterations
        document_lines = [{"content": f"Line {i}"} for i in range(300)]
        
        # Simulate the iterative processing that happens in get_sections
        all_sections = []
        start_line = 0
        max_iterations = 5
        
        for _ in range(max_iterations):
            doc_str, end_line = get_document_with_lines(document_lines, start_line, max_characters=100)
            
            # Simulate what the LLM would return
            new_sections = [
                DocumentSection(title=f"Section {i}", start_index=start_line + i*20)
                for i in range(3)
            ]
            
            # Validate and fix the sections
            fixed_sections = validate_and_fix_sections(new_sections, len(document_lines))
            
            if not fixed_sections:
                start_line = end_line + 1
                continue
            
            all_sections.extend(fixed_sections)
            
            if end_line >= len(document_lines) - 1:
                break
            
            if len(fixed_sections) > 1:
                # Start from the beginning of the last section to ensure proper overlap handling
                start_line = all_sections[-1].start_index
                all_sections.pop()
            else:
                start_line = end_line + 1
            
            if start_line >= len(document_lines):
                break
        
        # Get the final sections
        final_sections = get_sections_text(all_sections, document_lines)
        
        # Verify no sections extend beyond document bounds
        self.assertLess(final_sections[-1]["end"], len(document_lines))
        
        # Verify sections are continuous (no gaps)
        for i in range(len(final_sections) - 1):
            self.assertEqual(final_sections[i]["end"] + 1, final_sections[i + 1]["start"], 
                            f"Gap between sections {i} and {i+1}")
        
        # Verify all sections are within bounds
        for section in final_sections:
            self.assertGreaterEqual(section["start"], 0)
            self.assertLess(section["end"], len(document_lines))
            self.assertLess(section["start"], section["end"])

    def test_get_sections_overlap_handling(self):
        """Test that section overlaps are handled correctly"""
        document_lines = [{"content": f"Line {i}"} for i in range(100)]
        
        # Create overlapping sections
        sections = [
            DocumentSection(title="Section 1", start_index=0),
            DocumentSection(title="Section 2", start_index=40),  # Overlaps with Section 3
            DocumentSection(title="Section 3", start_index=30)   # Overlaps with Section 2
        ]
        
        fixed = validate_and_fix_sections(sections, document_length=len(document_lines))
        result = get_sections_text(fixed, document_lines)
        
        # Verify sections are properly ordered and don't overlap
        for i in range(len(result) - 1):
            self.assertLess(result[i]["end"], result[i + 1]["start"], 
                           f"Sections {i} and {i+1} overlap")

    def test_get_document_with_lines_bounds(self):
        """Test that get_document_with_lines respects document bounds"""
        document_lines = [{"content": f"Line {i}"} for i in range(50)]
        
        # Test starting from middle
        doc_str, end_line = get_document_with_lines(document_lines, start_line=25, max_characters=1000)
        self.assertLess(end_line, len(document_lines))
        
        # Test with start_line near end of document
        doc_str, end_line = get_document_with_lines(document_lines, start_line=45, max_characters=1000)
        self.assertLess(end_line, len(document_lines))
        
        # Test with start_line at end of document
        doc_str, end_line = get_document_with_lines(document_lines, start_line=49, max_characters=1000)
        self.assertEqual(end_line, len(document_lines) - 1)

    def test_section_continuity(self):
        """Test that sections remain continuous across multiple processing iterations"""
        # Create a document that will require multiple iterations
        document_lines = [{"content": f"Line {i}"} for i in range(200)]
        
        all_sections = []
        start_line = 0
        max_iterations = 3
        
        for _ in range(max_iterations):
            doc_str, end_line = get_document_with_lines(document_lines, start_line, max_characters=100)
            
            # Create some test sections
            new_sections = [
                DocumentSection(title=f"Section {i}", start_index=start_line + i*10)
                for i in range(3)
            ]
            
            # Validate and fix sections
            fixed_sections = validate_and_fix_sections(new_sections, len(document_lines))
            
            if not fixed_sections:
                start_line = end_line + 1
                continue
            
            all_sections.extend(fixed_sections)
            
            if end_line >= len(document_lines) - 1:
                break
            
            if len(fixed_sections) > 1:
                # Test the overlap handling logic
                start_line = all_sections[-1].start_index
                all_sections.pop()
            else:
                start_line = end_line + 1
            
            self.assertLess(start_line, len(document_lines), 
                           "start_line exceeded document length")
        
        # Convert to final sections
        final_sections = get_sections_text(all_sections, document_lines)
        
        # Verify final sections
        for section in final_sections:
            self.assertLess(section["end"], len(document_lines), 
                           f"Section end {section['end']} exceeds document length {len(document_lines)}")
            self.assertGreaterEqual(section["start"], 0, 
                                  f"Section start {section['start']} is negative")

if __name__ == "__main__":
    unittest.main()