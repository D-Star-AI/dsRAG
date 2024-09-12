from dsrag.database.vector.db import VectorDB
from dsrag.database.vector.types import VectorSearchResult, MetadataFilter
from typing import Optional
import os
import chromadb
import numpy as np


def format_metadata_filter(metadata_filter: MetadataFilter) -> dict:
    """
    Format the metadata filter to be used in the ChromaDB query method.

    Args:
        metadata_filter (dict): The metadata filter.

    Returns:
        dict: The formatted metadata filter.
    """

    field = metadata_filter["field"]
    operator = metadata_filter["operator"]
    value = metadata_filter["value"]

    operator_mapping = {
        "equals": "$eq",
        "not_equals": "$ne",
        "in": "$in",
        "not_in": "$nin",
        "greater_than": "$gt",
        "less_than": "$lt",
        "greater_than_equals": "$gte",
        "less_than_equals": "$lte",
    }

    formatted_operator = operator_mapping[operator]
    formatted_metadata_filter = {field: {formatted_operator: value}}
    
    return formatted_metadata_filter


class ChromaDB(VectorDB):

    def __init__(self, kb_id: str, storage_directory: str = "~/dsRAG"):
        self.kb_id = kb_id
        self.storage_directory = os.path.expanduser(storage_directory)
        self.vector_storage_path = os.path.join(
            self.storage_directory, "vector_storage"
        )
        self.client = chromadb.PersistentClient(path=self.vector_storage_path)
        self.collection = self.client.get_or_create_collection(
            kb_id, metadata={"hnsw:space": "cosine"}
        )

    def get_num_vectors(self):
        return self.collection.count()

    def add_vectors(self, vectors: list, metadata: list):

        # Convert NumPy arrays to lists
        vectors_as_lists = [vector.tolist() if isinstance(vector, np.ndarray) else vector for vector in vectors]
        try:
            assert len(vectors_as_lists) == len(metadata)
        except AssertionError:
            raise ValueError(
                "Error in add_vectors: the number of vectors and metadata items must be the same."
            )

        # Create the ids from the metadata, defined as {metadata["doc_id"]}_{metadata["chunk_index"]}
        ids = [f"{meta['doc_id']}_{meta['chunk_index']}" for meta in metadata]

        self.collection.add(embeddings=vectors_as_lists, metadatas=metadata, ids=ids)

    def search(self, query_vector, top_k=10, metadata_filter: Optional[MetadataFilter] = None) -> list[VectorSearchResult]:

        num_vectors = self.get_num_vectors()
        if num_vectors == 0:
            return []
            # raise ValueError('No vectors stored in the database.')

        if metadata_filter:
            formatted_metadata_filter = format_metadata_filter(metadata_filter)

        query_results = self.collection.query(
                query_embeddings=query_vector,
                n_results=top_k,
                include=["distances", "metadatas"],
                where=formatted_metadata_filter if metadata_filter else None
            )

        metadata = query_results["metadatas"][0]
        distances = query_results["distances"][0]

        results: list[VectorSearchResult] = []
        for _, (distance, metadata) in enumerate(zip(distances, metadata)):
            results.append(
                VectorSearchResult(
                    doc_id=metadata["doc_id"],
                    vector=None,
                    metadata=metadata,
                    similarity=1 - distance,
                )
            )

        results = sorted(results, key=lambda x: x["similarity"], reverse=True)

        return results

    def remove_document(self, doc_id: str):
        data = self.collection.get(where={"doc_id": doc_id})
        doc_ids = data["ids"]
        self.collection.delete(ids=doc_ids)

    def delete(self):
        self.client.delete_collection(name=self.kb_id)

    def to_dict(self):
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
        }
