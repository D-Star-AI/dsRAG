import cohere
import os

def rerank_search_results(query: str, search_results: list) -> list:
    """
    Use Cohere Rerank API to rerank the search results
    """
    cohere_api_key = os.environ['COHERE_API_KEY']
    co = cohere.Client(f'{cohere_api_key}')
    documents = [f"[{result['metadata']['chunk_header']}]\n{result['metadata']['chunk_text']}" for result in search_results]
    reranked_results = co.rerank(query, documents, model="rerank-english-v3.0")
    results = reranked_results.results
    reranked_indices = [result.index for result in results]
    print (reranked_indices)
    reranked_search_results = [search_results[i] for i in reranked_indices]
    return reranked_search_results