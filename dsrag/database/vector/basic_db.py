import pickle
from dsrag.database.vector.db import VectorDB
from typing import Sequence, Optional
from dsrag.database.vector.types import ChunkMetadata, Vector, VectorSearchResult
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np


class BasicVectorDB(VectorDB):
    def __init__(
        self, kb_id: str, storage_directory: str = "~/dsRAG", use_faiss: bool = True
    ) -> None:
        self.kb_id = kb_id
        self.storage_directory = storage_directory
        self.use_faiss = use_faiss
        self.vector_storage_path = os.path.join(
            self.storage_directory, "vector_storage", f"{kb_id}.pkl"
        )
        self.load()

    def add_vectors(
        self, vectors: Sequence[Vector], metadata: Sequence[ChunkMetadata]
    ) -> None:
        try:
            assert len(vectors) == len(metadata)
        except AssertionError:
            raise ValueError(
                "Error in add_vectors: the number of vectors and metadata items must be the same."
            )
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)
        self.save()

    def search(self, query_vector, top_k=10, metadata_filter: Optional[dict] = None) -> list[VectorSearchResult]:
        if not self.vectors:
            return []

        if self.use_faiss:
            return self.search_faiss(query_vector, top_k)

        similarities = cosine_similarity([query_vector], self.vectors)[0]
        indexed_similarities = sorted(
            enumerate(similarities), key=lambda x: x[1], reverse=True
        )
        results: list[VectorSearchResult] = []
        for i, similarity in indexed_similarities[:top_k]:
            result = VectorSearchResult(
                doc_id=None,
                vector=None,
                metadata=self.metadata[i],
                similarity=similarity,
            )
            results.append(result)
        return results

    def search_faiss(self, query_vector, top_k=10) -> list[VectorSearchResult]:
        from faiss.contrib.exhaustive_search import knn

        # Limit top_k to the number of vectors we have - Faiss doesn't automatically handle this
        top_k = min(top_k, len(self.vectors))

        # faiss expects 2D arrays of vectors
        vectors_array = (
            np.array(self.vectors).astype("float32").reshape(len(self.vectors), -1)
        )
        query_vector_array = np.array(query_vector).astype("float32").reshape(1, -1)

        _, I = knn(
            query_vector_array, vectors_array, top_k
        )  # I is a list of indices in the corpus_vectors array
        results: list[VectorSearchResult] = []
        for i in I[0]:
            result = VectorSearchResult(
                doc_id=None,
                vector=None,
                metadata=self.metadata[i],
                similarity=cosine_similarity([query_vector], [self.vectors[i]])[0][0],
            )
            results.append(result)
        return results

    def remove_document(self, doc_id):
        i = 0
        while i < len(self.metadata):
            if self.metadata[i]["doc_id"] == doc_id:
                del self.vectors[i]
                del self.metadata[i]
            else:
                i += 1
        self.save()

    def save(self):
        os.makedirs(
            os.path.dirname(self.vector_storage_path), exist_ok=True
        )  # Ensure the directory exists
        with open(self.vector_storage_path, "wb") as f:
            pickle.dump((self.vectors, self.metadata), f)

    def load(self):
        if os.path.exists(self.vector_storage_path):
            with open(self.vector_storage_path, "rb") as f:
                self.vectors, self.metadata = pickle.load(f)
        else:
            self.vectors = []
            self.metadata = []

    def delete(self):
        if os.path.exists(self.vector_storage_path):
            os.remove(self.vector_storage_path)

    def to_dict(self):
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
            "storage_directory": self.storage_directory,
            "use_faiss": self.use_faiss,
        }
