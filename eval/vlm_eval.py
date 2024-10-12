import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from dsrag.knowledge_base import KnowledgeBase

kb = KnowledgeBase(kb_id="mck_energy", exists_ok=True)
kb.delete()

file_path = "/Users/zach/Code/mck_energy.pdf"
kb = KnowledgeBase(kb_id="mck_energy")
kb.add_document(
    doc_id="mck_energy_report",
    file_path=file_path,
    document_title="McKinsey Energy Report",
    file_parsing_config={
        "use_vlm": True,
        "vlm_config": {
            "provider": "vertex_ai",
            "model": "gemini-1.5-pro-002",
            "project_id": os.environ["VERTEX_PROJECT_ID"],
            "location": "us-central1",
            "save_path": "zmcc/mck_energy"
        }
    }
)

"""
kb = KnowledgeBase(kb_id="mck_energy")

#query = "What is the McKinsey Energy Report about?"
query = "How does the oil and gas industry compare to other industries in terms of its value prop to employees?"

rse_params = {
    "minimum_value": 0.0,
    "irrelevant_chunk_penalty": 0.1,
}

search_results = kb.query(search_queries=[query], rse_params=rse_params, return_images=True)
print (search_results)
"""