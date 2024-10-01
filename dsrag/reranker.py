from abc import ABC, abstractmethod
import cohere
import voyageai
import os
from scipy.stats import beta
from typing import Optional


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
        base_url = os.environ.get("DSRAG_COHERE_BASE_URL", None)
        if base_url is not None:
            self.client = cohere.Client(api_key=cohere_api_key)
        else:
            self.client = cohere.Client(api_key=cohere_api_key)

    def transform(self, x):
        """
        transformation function to map the absolute relevance value to a value that is more uniformly distributed between 0 and 1
        - this is critical for the new version of RSE to work properly, because it utilizes the absolute relevance values to calculate the similarity scores
        """
        a, b = 0.4, 0.4  # These can be adjusted to change the distribution shape
        return beta.cdf(x, a, b)

    def rerank_search_results(self, query: str, search_results: list) -> list:
        """
        Use Cohere Rerank API to rerank the search results
        """
        documents = []
        for result in search_results:
            documents.append(f"{result['metadata']['chunk_header']}\n\n{result['metadata']['chunk_text']}")

        reranked_results = self.client.rerank(model=self.model, query=query, documents=documents)
        results = reranked_results.results
        reranked_indices = [result.index for result in results]
        reranked_similarity_scores = [result.relevance_score for result in results]
        reranked_search_results = [search_results[i] for i in reranked_indices]
        for i, result in enumerate(reranked_search_results):
            result['similarity'] = self.transform(reranked_similarity_scores[i])
        return reranked_search_results
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model
        })
        return base_dict
    
class VoyageReranker(Reranker):
    def __init__(self, model: str = "rerank-1"):
        self.model = model
        voyage_api_key = os.environ['VOYAGE_API_KEY']
        self.client = voyageai.Client(api_key=voyage_api_key)

    def transform(self, x):
        """
        transformation function to map the absolute relevance value to a value that is more uniformly distributed between 0 and 1
        - this is critical for the new version of RSE to work properly, because it utilizes the absolute relevance values to calculate the similarity scores
        """
        a, b = 0.5, 1.8  # These can be adjusted to change the distribution shape
        return beta.cdf(x, a, b)

    def rerank_search_results(self, query: str, search_results: list) -> list:
        """
        Use Voyage Rerank API to rerank the search results
        """
        documents = []
        for result in search_results:
            documents.append(f"{result['metadata']['chunk_header']}\n\n{result['metadata']['chunk_text']}")
        
        reranked_results = self.client.rerank(model=self.model, query=query, documents=documents)
        results = reranked_results.results
        reranked_indices = [result.index for result in results]
        reranked_similarity_scores = [result.relevance_score for result in results]
        reranked_search_results = [search_results[i] for i in reranked_indices]
        for i, result in enumerate(reranked_search_results):
            result['similarity'] = self.transform(reranked_similarity_scores[i])
        return reranked_search_results
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model
        })
        return base_dict
    
class NoReranker(Reranker):
    def __init__(self, ignore_absolute_relevance: bool = False):
        """
        - ignore_absolute_relevance: if True, the reranker will override the absolute relevance values and assign a default similarity score to each chunk. This is useful when using an embedding model where the absolute relevance values are not reliable or meaningful.
        """
        self.ignore_absolute_relevance = ignore_absolute_relevance

    def rerank_search_results(self, query: str, search_results: list) -> list:
        if self.ignore_absolute_relevance:
            for result in search_results:
                result['similarity'] = 0.8 # default similarity score (represents a moderately relevant chunk)
        return search_results

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'ignore_absolute_relevance': self.ignore_absolute_relevance,
        })
        return base_dict