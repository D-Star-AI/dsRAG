import sys
import os

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sprag.create_kb import create_kb_from_file
from sprag.knowledge_base import KnowledgeBase

def test_create_kb_from_file():
    cleanup() # delete the KnowledgeBase object if it exists so we can start fresh
    
    # Get the absolute path of the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the dataset file
    file_path = os.path.join(script_dir, "../data/levels_of_agi.pdf")
    
    kb_id = "levels_of_agi"
    kb = create_kb_from_file(kb_id, file_path)

    # verify that the document is in the chunk db
    assert len(kb.chunk_db.get_all_doc_ids()) == 1

    # run a query and verify results are returned
    search_queries = ["What are the levels of AGI?", "What is the highest level of AGI?"]
    segment_info = kb.query(search_queries)
    assert len(segment_info) > 0

    # delete the KnowledgeBase object
    kb.delete()

def cleanup():
    kb = KnowledgeBase(kb_id="levels_of_agi", exists_ok=True)
    kb.delete()

if __name__ == "__main__":
    test_create_kb_from_file()