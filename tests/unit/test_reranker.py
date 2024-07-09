import sys
import os
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dsrag.reranker import Reranker, CohereReranker, VoyageReranker, NoReranker


class TestReranker(unittest.TestCase):
    def test_rerank_search_results_cohere(self):
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
        self.assertEqual(len(reranked_search_results), 2)
        self.assertEqual(reranked_search_results[0]["metadata"]["chunk_text"], "Hello, world!")
        self.assertEqual(reranked_search_results[1]["metadata"]["chunk_text"], "Goodbye, world!")

    def test_rerank_search_results_voyage(self):
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
        reranker = VoyageReranker()
        reranked_search_results = reranker.rerank_search_results(query, search_results)
        self.assertEqual(len(reranked_search_results), 2)
        self.assertEqual(reranked_search_results[0]["metadata"]["chunk_text"], "Hello, world!")
        self.assertEqual(reranked_search_results[1]["metadata"]["chunk_text"], "Goodbye, world!")

    def test_save_and_load_from_dict(self):
        reranker = CohereReranker(model="embed-english-v3.0")
        config = reranker.to_dict()
        reranker_instance = Reranker.from_dict(config)
        self.assertEqual(reranker_instance.model, "embed-english-v3.0")

    def test_no_reranker(self):
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
        self.assertEqual(len(reranked_search_results), 2)
        self.assertEqual(reranked_search_results[0]["metadata"]["chunk_text"], "Hello, world!")
        self.assertEqual(reranked_search_results[1]["metadata"]["chunk_text"], "Goodbye, world!")


if __name__ == "__main__":
    unittest.main()