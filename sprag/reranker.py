from abc import ABC, abstractmethod
import cohere
import os

class Reranker(ABC):
    @abstractmethod
    def rerank_search_results(self, query: str, search_results: list) -> list:
        pass

class CohereReranker(Reranker):
    def __init__(self, model: str = "rerank-english-v3.0"):
        self.model = model
        cohere_api_key = os.environ['COHERE_API_KEY']
        self.client = cohere.Client(f'{cohere_api_key}')

    def rerank_search_results(self, query: str, search_results: list) -> list:
        """
        Use Cohere Rerank API to rerank the search results
        """
        documents = [f"[{result['metadata']['chunk_header']}]\n{result['metadata']['chunk_text']}" for result in search_results]
        reranked_results = self.client.rerank(query, documents, model=self.model)
        results = reranked_results.results
        reranked_indices = [result.index for result in results]
        reranked_search_results = [search_results[i] for i in reranked_indices]
        return reranked_search_results