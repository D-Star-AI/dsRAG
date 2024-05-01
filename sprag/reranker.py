from abc import ABC, abstractmethod
import cohere
import os

class Reranker(ABC):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def to_dict(self):
        return {
            'subclass_name': self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config):
        subclass_name = config.pop('subclass_name', None)  # Remove subclass_name from config
        subclass = cls.subclasses.get(subclass_name)
        if subclass:
            return subclass(**config)  # Pass the modified config without subclass_name
        else:
            raise ValueError(f"Unknown subclass: {subclass_name}")

    @abstractmethod
    def rerank_search_results(self, query: str, search_results: list) -> list:
        pass

class CohereReranker(Reranker):
    def __init__(self, model: str = "rerank-english-v3.0"):
        self.model = model
        cohere_api_key = os.environ['CO_API_KEY']
        self.client = cohere.Client(f'{cohere_api_key}')

    def rerank_search_results(self, query: str, search_results: list) -> list:
        """
        Use Cohere Rerank API to rerank the search results
        """
        documents = [f"[{result['metadata']['chunk_header']}]\n{result['metadata']['chunk_text']}" for result in search_results]
        reranked_results = self.client.rerank(model=self.model, query=query, documents=documents)
        results = reranked_results.results
        reranked_indices = [result.index for result in results]
        reranked_search_results = [search_results[i] for i in reranked_indices]
        return reranked_search_results
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model,
        })
        return base_dict
    
class NoReranker(Reranker):
    def rerank_search_results(self, query: str, search_results: list) -> list:
        return search_results

    def to_dict(self):
        return super().to_dict()