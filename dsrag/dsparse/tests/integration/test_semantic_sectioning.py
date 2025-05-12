import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dsparse.sectioning_and_chunking.semantic_sectioning import get_sections_from_str_parallel

class TestSemanticSectioning(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        # Sample text with clear sections for testing - same as original test
        self.test_document = """Introduction
Artificial intelligence has revolutionized many fields in recent years, fundamentally changing how we approach complex problems.
This paper examines the impact of AI on healthcare, particularly in diagnostic and treatment planning applications.
The integration of AI systems into clinical workflows has shown promising results across multiple specialties.
Recent advances in machine learning algorithms and computational power have made it possible to process vast amounts of medical data efficiently.
Despite these advances, questions remain about the optimal implementation of AI in healthcare settings.

Methods
We conducted a systematic review of 100 papers published between 2019 and 2024 in major medical and computer science journals.
The papers were analyzed using both quantitative and qualitative methods to assess the impact of AI implementations.
Our analysis framework included metrics for diagnostic accuracy, clinical workflow efficiency, and patient outcomes.
We specifically focused on studies that implemented deep learning models in radiology, pathology, and clinical decision support.
The selected papers were independently reviewed by three researchers using a standardized evaluation protocol.
Statistical analysis was performed using R version 4.2.0, with significance set at p < 0.05.

Results
Our analysis showed significant improvements in diagnostic accuracy across multiple medical specialties.
AI systems demonstrated 95% accuracy in image recognition tasks, particularly in radiology and dermatology applications.
Implementation of AI-powered clinical decision support tools reduced diagnostic time by an average of 37%.
Cost-benefit analyses revealed a positive return on investment within 18 months of implementation.
Patient satisfaction scores increased by 22% in facilities using AI-assisted diagnostic tools.
Notably, integration challenges were reported in 45% of implementations, primarily related to workflow adaptation.
The highest success rates were observed in facilities that implemented comprehensive staff training programs.

Conclusion
AI has shown great promise in healthcare applications, particularly in diagnostic and decision support roles.
Future research should focus on implementation challenges and strategies for seamless integration into clinical workflows.
Our findings suggest that successful AI implementation requires a balanced approach considering technical, organizational, and human factors.
Standardization of AI validation protocols and implementation guidelines emerges as a critical need in the field.
The potential for AI to improve healthcare delivery remains high, but careful consideration must be given to practical implementation challenges."""

        # Longer document to test parallelization
        self.longer_document = self.test_document * 3
        self.test_document_short = "This is a short document."

    def _validate_sections(self, sections, document_lines):
        """Helper method to validate section structure"""
        self.assertTrue(len(sections) > 0)
        self.assertEqual(type(sections), list)
        
        # Validate section types
        for section in sections:
            # Validate section has required keys
            self.assertIn('title', section)
            self.assertIn('content', section)
            self.assertIn('start', section)
            self.assertIn('end', section)
            
            # Validate content
            self.assertTrue(len(section['title']) > 0)
            self.assertTrue(len(section['content']) > 0)
            self.assertIsInstance(section['start'], int)
            self.assertIsInstance(section['end'], int)
            
            # Validate section boundaries
            self.assertGreaterEqual(section['start'], 0)  # First line is 0
            self.assertLess(section['end'], len(document_lines))
            self.assertLess(section['start'], section['end'])

    def test_openai_semantic_sectioning(self):
        semantic_sectioning_config = {
            "use_semantic_sectioning": True,
            "llm_provider": "openai",
            "model": "gpt-4o-mini",
            "language": "en",
            "max_concurrent_llm_calls": 3  # For parallel processing
        }

        chunking_config = { 
            "min_length_for_chunking": 1000
        }
        
        sections, document_lines = get_sections_from_str_parallel(
            document=self.longer_document,
            max_characters_per_window=5000,  # Smaller windows to test parallelization
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config=chunking_config
        )
        
        self._validate_sections(sections, document_lines)

    def test_anthropic_semantic_sectioning(self):
        semantic_sectioning_config = {
            "use_semantic_sectioning": True,
            "llm_provider": "anthropic",
            "model": "claude-3-5-haiku-latest",
            "language": "en",
            "max_concurrent_llm_calls": 3  # For parallel processing
        }

        chunking_config = { 
            "min_length_for_chunking": 1000
        }
        
        sections, document_lines = get_sections_from_str_parallel(
            document=self.longer_document,
            max_characters_per_window=5000,  # Smaller windows to test parallelization
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config=chunking_config
        )
        
        self._validate_sections(sections, document_lines)

    def test_gemini_semantic_sectioning(self):
        semantic_sectioning_config = {
            "use_semantic_sectioning": True,
            "llm_provider": "gemini",
            "model": "gemini-2.0-flash",
            "language": "en",
            "max_concurrent_llm_calls": 3  # For parallel processing
        }

        chunking_config = { 
            "min_length_for_chunking": 1000
        }
        
        sections, document_lines = get_sections_from_str_parallel(
            document=self.longer_document,
            max_characters_per_window=5000,  # Smaller windows to test parallelization
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config=chunking_config
        )
        
        self._validate_sections(sections, document_lines)

    def test_no_semantic_sectioning(self):
        semantic_sectioning_config = {
            "use_semantic_sectioning": False,
        }

        chunking_config = {
            "min_length_for_chunking": 1000
        }
        
        sections, document_lines = get_sections_from_str_parallel(
            document=self.test_document,
            max_characters_per_window=5000,
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config=chunking_config
        )
        
        # Should return a single section containing the entire document
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]['start'], 0)
        self.assertEqual(sections[0]['end'], len(document_lines) - 1)
        self.assertEqual(sections[0]['content'], self.test_document)

    def test_short_document(self):
        semantic_sectioning_config = {
            "use_semantic_sectioning": True,
            "max_concurrent_llm_calls": 3
        }

        chunking_config = {
            "min_length_for_chunking": 1000
        }

        sections, document_lines = get_sections_from_str_parallel(
            document=self.test_document_short,
            max_characters_per_window=5000,
            semantic_sectioning_config=semantic_sectioning_config,
            chunking_config=chunking_config
        )

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]['start'], 0)
        self.assertEqual(sections[0]['end'], len(document_lines) - 1)
        self.assertEqual(sections[0]['content'], self.test_document_short)

    def test_concurrency_levels(self):
        """Test different concurrency levels"""
        # This test specifically tests that different max_concurrent_llm_calls values work
        
        for concurrency in [1, 2, 4]:
            semantic_sectioning_config = {
                "use_semantic_sectioning": True,
                "llm_provider": "openai",
                "model": "gpt-4o-mini",
                "language": "en",
                "max_concurrent_llm_calls": concurrency
            }

            chunking_config = { 
                "min_length_for_chunking": 1000
            }
            
            sections, document_lines = get_sections_from_str_parallel(
                document=self.longer_document,
                max_characters_per_window=3000,  # Small windows to force multiple chunks
                semantic_sectioning_config=semantic_sectioning_config,
                chunking_config=chunking_config
            )
            
            self._validate_sections(sections, document_lines)
            
            # Output concurrency level and number of sections found
            print(f"Concurrency level {concurrency} produced {len(sections)} sections")

if __name__ == "__main__":
    unittest.main()