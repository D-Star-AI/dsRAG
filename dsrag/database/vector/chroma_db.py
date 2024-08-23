from dsrag.database.vector.db import VectorDB
from dsrag.database.vector.types import VectorSearchResult

import os
import chromadb


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

    def add_vectors(self, vectors, metadata):
        try:
            assert len(vectors) == len(metadata)
        except AssertionError:
            raise ValueError(
                "Error in add_vectors: the number of vectors and metadata items must be the same."
            )

        # Create the ids from the metadata, defined as {metadata["doc_id"]}_{metadata["chunk_index"]}
        ids = [f"{meta['doc_id']}_{meta['chunk_index']}" for meta in metadata]

        self.collection.add(embeddings=vectors, metadatas=metadata, ids=ids)

    def search(self, query_vector, top_k=10) -> list[VectorSearchResult]:

        num_vectors = self.get_num_vectors()
        if num_vectors == 0:
            return []
            # raise ValueError('No vectors stored in the database.')

        query_results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["distances", "metadatas"],
        )

        metadata = query_results["metadatas"][0]
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

    def remove_document(self, doc_id):
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
