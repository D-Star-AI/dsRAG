import os
import sys
import unittest
import json
from unittest.mock import patch, Mock, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dsparse.sectioning_and_chunking.ss_parallel import (
    get_document_text_for_window,
    validate_and_fix_window_sections,
    split_long_line,
    elements_to_lines,
    str_to_lines,
    pages_to_lines,
    DocumentSection,
    get_sections_text,
    create_document_windows,
    merge_sections_across_windows,
    validate_and_fix_global_sections,
    parallel_get_sections,
    get_sections_from_str_parallel,
)

class TestParallelSemanticSectioning(unittest.TestCase):
    def setUp(self):
        # Create sample document lines for testing
        self.document_lines = [{"content": f"Line {i}"} for i in range(100)]
        
        # Sample document string
        self.sample_document = "\n".join([f"Line {i}" for i in range(100)])
        
        # Data path for loading test data
        self.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
        
    def test_document_text_for_window(self):
        """Test that window text is generated correctly with proper line numbers"""
        window_text = get_document_text_for_window(self.document_lines, 10, 20)
        
        # Verify the first and last line have correct line numbers
        lines = window_text.strip().split("\n")
        self.assertTrue(lines[0].startswith("[10]"))
        self.assertTrue(lines[-1].startswith("[20]"))
        
        # Verify correct number of lines
        self.assertEqual(len(lines), 11)  # 10-20 inclusive = 11 lines
        
        # Test boundary conditions
        window_text = get_document_text_for_window(self.document_lines, 90, 110)
        lines = window_text.strip().split("\n")
        self.assertTrue(lines[0].startswith("[90]"))
        # Should be truncated to document length
        self.assertTrue(lines[-1].startswith("[99]"))
    
    def test_validate_and_fix_window_sections(self):
        """Test that window section validation works correctly"""
        # Test with valid sections
        window_sections = [
            DocumentSection(title="Window Section 1", start_index=10),
            DocumentSection(title="Window Section 2", start_index=15),
            DocumentSection(title="Window Section 3", start_index=18)
        ]
        
        fixed = validate_and_fix_window_sections(
            window_sections, 
            window_start_line=10, 
            window_end_line=20,
            document_length=100
        )
        
        # First section should start at window start
        self.assertEqual(fixed[0].start_index, 10)
        
        # Test with out-of-order sections
        window_sections = [
            DocumentSection(title="Window Section 2", start_index=15),
            DocumentSection(title="Window Section 1", start_index=10),
            DocumentSection(title="Window Section 3", start_index=18)
        ]
        
        fixed = validate_and_fix_window_sections(
            window_sections, 
            window_start_line=10, 
            window_end_line=20,
            document_length=100
        )
        
        # Sections should be ordered
        self.assertEqual(fixed[0].title, "Window Section 1")
        self.assertEqual(fixed[1].title, "Window Section 2")
        self.assertEqual(fixed[2].title, "Window Section 3")
        
        # Test with section beyond window
        window_sections = [
            DocumentSection(title="Window Section 1", start_index=10),
            DocumentSection(title="Window Section 2", start_index=25)  # Beyond window end
        ]
        
        fixed = validate_and_fix_window_sections(
            window_sections, 
            window_start_line=10, 
            window_end_line=20,
            document_length=100
        )
        
        # Section beyond window should be removed
        self.assertEqual(len(fixed), 1)
        
        # Test with empty sections list
        fixed = validate_and_fix_window_sections(
            [], 
            window_start_line=10, 
            window_end_line=20,
            document_length=100
        )
        
        # Should create a default section
        self.assertEqual(len(fixed), 1)
        self.assertEqual(fixed[0].start_index, 10)
    
    def test_create_document_windows(self):
        """Test window creation works correctly"""
        # Create a test document with varying line lengths
        lines_with_varying_lengths = []
        for i in range(100):
            # Some lines are long, some are short
            content = f"Line {i}" + " extra content" * (i % 5)
            lines_with_varying_lengths.append({"content": content})
        
        # Test with small window size to force multiple windows
        windows = create_document_windows(lines_with_varying_lengths, max_characters_per_window=500)
        
        # Should have multiple windows
        self.assertTrue(len(windows) > 1)
        
        # Windows should cover the entire document
        self.assertEqual(windows[0][0], 0)  # First window starts at beginning
        self.assertEqual(windows[-1][1], 99)  # Last window ends at end
        
        # Windows should not overlap
        for i in range(len(windows) - 1):
            self.assertEqual(windows[i][1] + 1, windows[i+1][0], 
                           f"Window {i} and {i+1} overlap or have gap")
        
        # Test with empty document
        empty_windows = create_document_windows([], max_characters_per_window=500)
        self.assertEqual(len(empty_windows), 0)
    
    def test_merge_sections_across_windows(self):
        """Test section merging across windows"""
        # Create sections from 3 windows
        window1_sections = [
            DocumentSection(title="Window 1 Section 1", start_index=0),
            DocumentSection(title="Window 1 Section 2", start_index=5)
        ]
        
        window2_sections = [
            DocumentSection(title="Window 2 Section 1", start_index=10),
            DocumentSection(title="Window 2 Section 2", start_index=15)
        ]
        
        window3_sections = [
            DocumentSection(title="Window 3 Section 1", start_index=20),
            DocumentSection(title="Window 3 Section 2", start_index=25)
        ]
        
        all_window_sections = [window1_sections, window2_sections, window3_sections]
        
        merged_sections = merge_sections_across_windows(all_window_sections)
        
        # Sections after merging: W1S1, merged W1S2/W2S1, W2S2, merged W2S2/W3S1, W3S2
        # The number of sections should be 5 since we have 3 windows with 2 sections each
        # and each adjacent window pair gets 1 section merged, so: 6 - 1 = 5 sections
        self.assertEqual(len(merged_sections), 5)
        
        # First section should be preserved from window 1
        self.assertEqual(merged_sections[0].title, "Window 1 Section 1")
        
        # Last section from window 1 should be merged with first section from window 2
        self.assertTrue("Window 1 Section 2 / Window 2 Section 1" in merged_sections[1].title)
        
        # Test with empty window sections list
        merged_empty = merge_sections_across_windows([])
        self.assertEqual(len(merged_empty), 0)
        
        # Test with single window
        merged_single = merge_sections_across_windows([window1_sections])
        self.assertEqual(len(merged_single), 2)
        self.assertEqual(merged_single[0].title, "Window 1 Section 1")
    
    def test_validate_and_fix_global_sections(self):
        """Test global section validation"""
        # Create sections for testing
        sections = [
            DocumentSection(title="Section 1", start_index=10),  # Should be adjusted to 0
            DocumentSection(title="Section 2", start_index=20),
            DocumentSection(title="Section 3", start_index=30)
        ]
        
        fixed = validate_and_fix_global_sections(sections, document_length=100)
        
        # First section should start at 0
        self.assertEqual(fixed[0].start_index, 0)
        
        # Test with empty sections list
        fixed_empty = validate_and_fix_global_sections([], document_length=100)
        self.assertEqual(len(fixed_empty), 1)
        self.assertEqual(fixed_empty[0].start_index, 0)
        
        # Test with sections beyond document bounds
        beyond_bounds = [
            DocumentSection(title="Section 1", start_index=0),
            DocumentSection(title="Section 2", start_index=110)  # Beyond document length
        ]
        
        fixed_bounds = validate_and_fix_global_sections(beyond_bounds, document_length=100)
        self.assertEqual(len(fixed_bounds), 1)  # Section beyond bounds should be removed
    
    @patch('dsparse.sectioning_and_chunking.ss_parallel.get_structured_document_for_window')
    def test_parallel_get_sections(self, mock_get_structured_doc):
        """Test parallel processing of document sections"""
        # Mock the LLM call to return predetermined sections
        mock_response = Mock()
        mock_response.sections = [
            DocumentSection(title="Test Section 1", start_index=0),
            DocumentSection(title="Test Section 2", start_index=5)
        ]
        mock_get_structured_doc.return_value = mock_response
        
        # Process document with mocked LLM calls
        sections = parallel_get_sections(
            document_lines=self.document_lines,
            max_characters_per_window=500,
            llm_provider="openai",
            model="gpt-4o-mini",
            language="en",
            max_concurrent_llm_calls=2
        )
        
        # Should have produced sections
        self.assertTrue(len(sections) > 0)
        
        # Each section should have title and content
        for section in sections:
            self.assertIn("title", section)
            self.assertIn("content", section)
            self.assertIn("start", section)
            self.assertIn("end", section)
    
    def test_split_long_line(self):
        """Test splitting long lines"""
        # Short line doesn't need splitting
        short_line = "This is a short line"
        self.assertEqual(split_long_line(short_line, max_line_length=50), [short_line])
        
        # Long line needs splitting
        long_line = "This is a very long line that needs to be split into multiple lines because it exceeds the maximum length"
        split_lines = split_long_line(long_line, max_line_length=20)
        
        # Each line should be under max length
        self.assertTrue(all(len(line) <= 20 for line in split_lines))
        
        # Combined content should match original
        self.assertEqual(" ".join(split_lines), long_line)
    
    def test_elements_to_lines(self):
        """Test converting elements to lines"""
        elements = [
            {"type": "Title", "content": "Test Document"},
            {"type": "NarrativeText", "content": "This is a test.\nThis is another line."},
            {"type": "Table", "content": "Table content"},
            {"type": "Image", "content": "Image description"}
        ]
        
        # Test with no exclusions
        lines = elements_to_lines(elements, exclude_elements=[], visual_elements=["Image"])
        
        # Should have visual and non-visual elements
        self.assertTrue(any(line["is_visual"] for line in lines))
        self.assertTrue(any(not line["is_visual"] for line in lines))
        
        # Test with exclusions
        lines = elements_to_lines(elements, exclude_elements=["Table"], visual_elements=["Image"])
        self.assertFalse(any(line["element_type"] == "Table" for line in lines))
    
    def test_str_to_lines(self):
        """Test converting string to lines"""
        text = "Line 1\nLine 2\nA very long line that should be split into multiple lines"
        lines = str_to_lines(text, max_line_length=20)
        
        # Each line should be under max length
        self.assertTrue(all(len(line["content"]) <= 20 for line in lines))
        
        # All lines should be narrative text
        self.assertTrue(all(line["element_type"] == "NarrativeText" for line in lines))
    
    def test_pages_to_lines(self):
        """Test converting pages to lines"""
        pages = [
            "Page 1 content\nMore content",
            "Page 2 content\nEven more content"
        ]
        lines = pages_to_lines(pages)
        
        # Lines should have page numbers
        self.assertEqual(lines[0]["page_number"], 1)
        self.assertEqual(lines[2]["page_number"], 2)
    
    @patch('dsparse.sectioning_and_chunking.ss_parallel.parallel_get_sections')
    def test_get_sections_from_str_parallel(self, mock_parallel_get_sections):
        """Test the main entry point function with string input"""
        # Mock the parallel processing function
        mock_parallel_get_sections.return_value = [{"title": "Test", "content": "Content", "start": 0, "end": 10}]
        
        # Use the function with a string input
        sections, document_lines = get_sections_from_str_parallel(
            document=self.sample_document,
            max_characters_per_window=500,
            semantic_sectioning_config={"use_semantic_sectioning": True},
            chunking_config={}
        )
        
        # Should have at least one section
        self.assertTrue(len(sections) > 0)
        
        # Document lines should be populated
        self.assertTrue(len(document_lines) > 0)
        
        # Test with semantic sectioning disabled
        sections, document_lines = get_sections_from_str_parallel(
            document=self.sample_document,
            semantic_sectioning_config={"use_semantic_sectioning": False}
        )
        
        # Should have a single section containing the entire document
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["start"], 0)
        self.assertEqual(sections[0]["end"], len(document_lines) - 1)

if __name__ == "__main__":
    unittest.main()