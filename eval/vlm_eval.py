import os
import sys
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from dsrag.knowledge_base import KnowledgeBase


def create_kb_and_add_document():
    kb = KnowledgeBase(kb_id="mck_energy_first_5", exists_ok=True)
    kb.delete()

    file_path = "/Users/zach/Code/test_docs/mck_energy_first_5_pages.pdf"
    kb = KnowledgeBase(kb_id="mck_energy")
    kb.add_document(
        doc_id="mck_energy_report",
        file_path=file_path,
        document_title="McKinsey Energy Report",
        file_parsing_config={
            "use_vlm": True,
            "vlm_config": {
                "provider": "vertex_ai",
                "model": "gemini-1.5-flash-002",
                "project_id": os.environ["VERTEX_PROJECT_ID"],
                "location": "us-central1",
            }
        }
    )

    return kb

def get_response(user_input: str, search_results: List[Dict]):
    """
    Given a user input and potentially multimodal search results, generate a response
    """
    pass

# create or load KB
kb = create_kb_and_add_document()

"""
kb = KnowledgeBase(kb_id="mck_energy")

# print the number of chunks in the KB
num_chunks = len(kb.chunk_db.data["mck_energy_report"])
print(f"Number of chunks in the KB: {num_chunks}")

#query = "What is the McKinsey Energy Report about?"
query = "How does the oil and gas industry compare to other industries in terms of its value prop to employees?"

rse_params = {
    "minimum_value": 0.0,
    "irrelevant_chunk_penalty": 0.1,
}

search_results = kb.query(search_queries=[query], rse_params=rse_params, return_mode="page_images")
print (search_results)
"""