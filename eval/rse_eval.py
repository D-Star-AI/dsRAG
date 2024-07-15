import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from dsrag.create_kb import create_kb_from_file
from dsrag.knowledge_base import KnowledgeBase
from dsrag.reranker import NoReranker, CohereReranker, VoyageReranker
from dsrag.embedding import OpenAIEmbedding, CohereEmbedding
from dsrag.document_parsing import extract_text_from_pdf

 
def test_create_kb_from_file():
    cleanup() # delete the KnowledgeBase object if it exists so we can start fresh
    
    # Get the absolute path of the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the dataset file
    file_path = os.path.join(script_dir, "../tests/data/levels_of_agi.pdf")
    
    kb_id = "levels_of_agi"
    #kb = create_kb_from_file(kb_id, file_path)
    kb = KnowledgeBase(kb_id=kb_id, exists_ok=False)
    text = extract_text_from_pdf(file_path)
    kb.add_document(file_path, text)

def cleanup():
    kb = KnowledgeBase(kb_id="levels_of_agi", exists_ok=True)
    kb.delete()

"""
This script is for doing qualitative evaluation of RSE
"""


if __name__ == "__main__":
    """
    # create the KnowledgeBase
    try:
        test_create_kb_from_file()
    except ValueError as e:
        print(e)
    """

    rse_params = {
        'max_length': 20,
        'overall_max_length': 30,
        'minimum_value': 0.5,
        'irrelevant_chunk_penalty': 0.2,
        'overall_max_length_extension': 5,
        'decay_rate': 30,
        'top_k_for_document_selection': 7,
    }

    #reranker = NoReranker(ignore_absolute_relevance=True)
    reranker = CohereReranker()
    #reranker = VoyageReranker()
    
    # load the KnowledgeBase and query it
    kb = KnowledgeBase(kb_id="levels_of_agi", exists_ok=True, reranker=reranker)
    #print (kb.chunk_db.get_all_doc_ids())

    #search_queries = ["What are the levels of AGI?"]
    #search_queries = ["Who is the president of the United States?"]
    #search_queries = ["AI"]
    #search_queries = ["What is the difference between AGI and ASI?"]
    #search_queries = ["How does autonomy factor into AGI?"]
    #search_queries = ["Self-driving cars"]
    search_queries = ["Methodology for determining levels of AGI"]
    #search_queries = ["Principles for defining levels of AGI"]
    #search_queries = ["What is Autonomy Level 3"]
    #search_queries = ["Use of existing AI benchmarks like Big-bench and HELM"]
    #search_queries = ["Introduction", "Conclusion"]
    #search_queries = ["References"]

    relevant_segments = kb.query(search_queries=search_queries, rse_params=rse_params)
    
    print ()
    for segment in relevant_segments:
        print (len(segment["text"]))
        #print (segment["score"])
        #print (segment["text"])
        print ("---\n")