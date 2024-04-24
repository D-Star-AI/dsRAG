import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from sprag.reranker import CohereReranker, Reranker, NoReranker

def test_rerank_search_results():
    query = "Hello, world!"
    search_results = [
        {
            "metadata": {
                "chunk_header": "",
                "chunk_text": "Hello, world!"
            }
        },
        {
            "metadata": {
                "chunk_header": "",
                "chunk_text": "Goodbye, world!"
            }
        }
    ]
    reranker = CohereReranker()
    reranked_search_results = reranker.rerank_search_results(query, search_results)
    assert len(reranked_search_results) == 2
    assert reranked_search_results[0]["metadata"]["chunk_text"] == "Hello, world!"
    assert reranked_search_results[1]["metadata"]["chunk_text"] == "Goodbye, world!"

def test_save_and_load_from_dict():
    reranker = CohereReranker(model="embed-english-v3.0")
    config = reranker.to_dict()
    reranker_instance = Reranker.from_dict(config)
    assert reranker_instance.model == "embed-english-v3.0"

def test_no_reranker():
    reranker = NoReranker()
    query = "Hello, world!"
    search_results = [
        {
            "metadata": {
                "chunk_header": "",
                "chunk_text": "Hello, world!"
            }
        },
        {
            "metadata": {
                "chunk_header": "",
                "chunk_text": "Goodbye, world!"
            }
        }
    ]
    reranked_search_results = reranker.rerank_search_results(query, search_results)
    assert len(reranked_search_results) == 2
    assert reranked_search_results[0]["metadata"]["chunk_text"] == "Hello, world!"
    assert reranked_search_results[1]["metadata"]["chunk_text"] == "Goodbye, world!"

if __name__ == "__main__":
    # run tests
    test_rerank_search_results()
    test_save_and_load_from_dict()
    test_no_reranker()