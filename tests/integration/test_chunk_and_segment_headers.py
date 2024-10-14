import sys
import os
import unittest

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dsrag.document_parsing import extract_text_from_pdf
from dsrag.knowledge_base import KnowledgeBase


class TestChunkAndSegmentHeaders(unittest.TestCase):    
    def test_all_context_included(self):
        self.cleanup() # delete the KnowledgeBase object if it exists so we can start fresh
        
        # Get the absolute path of the script file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the dataset file
        file_path = os.path.join(script_dir, "../data/levels_of_agi.pdf")

        # extract text from the pdf
        text, _ = extract_text_from_pdf(file_path)
        
        kb_id = "levels_of_agi"
        kb = KnowledgeBase(kb_id=kb_id, exists_ok=False)

        # set configuration for auto context and semantic sectioning and add the document to the KnowledgeBase
        auto_context_config = {
            'use_generated_title': True,
            'get_document_summary': True,
            'get_section_summaries': True,
        }
        semantic_sectioning_config = {
            'use_semantic_sectioning': True,
        }
        kb.add_document(doc_id="doc_1", text=text, auto_context_config=auto_context_config, semantic_sectioning_config=semantic_sectioning_config)

        # verify that the chunk headers are as expected by running searches and looking at the chunk_header in the results
        search_query = "What are the Levels of AGI and why were they defined this way?"
        search_results = kb.search(search_query, top_k=1)
        chunk_header = search_results[0]['metadata']['chunk_header']
        print (f"\nCHUNK HEADER\n{chunk_header}")
        
        # verify that the segment headers are as expected
        segment_header = kb.get_segment_header(doc_id="doc_1", chunk_index=0)

        assert "Section context" in chunk_header
        assert "AGI" in segment_header

        # delete the KnowledgeBase object
        kb.delete()

    def test_no_context_included(self):
        self.cleanup() # delete the KnowledgeBase object if it exists so we can start fresh
        
        # Get the absolute path of the script file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the dataset file
        file_path = os.path.join(script_dir, "../data/levels_of_agi.pdf")

        # extract text from the pdf
        text, _ = extract_text_from_pdf(file_path)
        
        kb_id = "levels_of_agi"
        kb = KnowledgeBase(kb_id=kb_id, exists_ok=False)

        # set configuration for auto context and semantic sectioning and add the document to the KnowledgeBase
        auto_context_config = {
            'use_generated_title': False,
            'get_document_summary': False,
            'get_section_summaries': False,
        }
        semantic_sectioning_config = {
            'use_semantic_sectioning': False,
        }
        kb.add_document(doc_id="doc_1", text=text, auto_context_config=auto_context_config, semantic_sectioning_config=semantic_sectioning_config)

        # verify that the chunk headers are as expected by running searches and looking at the chunk_header in the results
        search_query = "What are the Levels of AGI and why were they defined this way?"
        search_results = kb.search(search_query, top_k=1)
        chunk_header = search_results[0]['metadata']['chunk_header']
        print (f"\nCHUNK HEADER\n{chunk_header}")
        
        # verify that the segment headers are as expected
        segment_header = kb.get_segment_header(doc_id="doc_1", chunk_index=0)

        assert "Section context" not in chunk_header
        assert "AGI" not in segment_header

        # delete the KnowledgeBase object
        kb.delete()
        

    def cleanup(self):
        kb = KnowledgeBase(kb_id="levels_of_agi", exists_ok=True)
        kb.delete()

if __name__ == "__main__":
    unittest.main()