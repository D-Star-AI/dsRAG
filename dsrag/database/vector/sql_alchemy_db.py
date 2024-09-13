from typing import Any, Sequence
from dsrag.database.vector.types import (
    ChunkEmbedding,
    ChunkMetadata,
    Vector,
    VectorSearchResult,
)
from .db import VectorDB


class PostgresVectorDB(VectorDB):
    """
    An implementation of the VectorDB interface for Postgres.

    This class provides methods for adding, removing, and searching for vectorized data
    within a Postgres instance.
    """

    def __init__(
        self,
        # Needs to be imported from SQLAlchemy
        db: Session,
        kb_id: str,
        init_timeout: int = 2,
        query_timeout: int = 45,
        insert_timeout: int = 120,
    ):
        self.kb_id = kb_id
        self.init_timeout = init_timeout
        self.query_timeout = query_timeout
        self.insert_timeout = insert_timeout
        self.db = db

    def add_vectors(
        self, vectors: Sequence[Vector], metadata: Sequence[ChunkMetadata]
    ) -> None:
        """
        Adds a list of vectors with associated metadata to Postgres Vector DB.

        Args:
            vectors: A list of vector embeddings.
            metadata: A list of dictionaries containing metadata for each vector.

        Raises:
            ValueError: If the number of vectors and metadata items do not match.
        """
        if len(vectors) != len(metadata):
            raise ValueError("The number of vectors and metadata items must match.")

        for vector, meta in zip(vectors, metadata, strict=False):
            doc_id = meta.get("doc_id", "")
            if not doc_id:
                raise ValueError("Metadata must contain a 'doc_id' key.")

            embedding = ChunkEmbedding(
                kb_id=self.kb_id,
                transcript_id=doc_id,
                vector=vector,
                chunk_metadata=meta,
            )
            self.db.add(embedding)
        self.db.commit()

    def remove_document(self, doc_id: str):
        """
        Removes a document (data object) from Postgres Vector DB.

        Args:
            doc_id: The UUID of the document to remove.
        """
        self.db.query(ChunkEmbedding).filter(
            ChunkEmbedding.chunk_metadata["doc_id"] == doc_id,
            ChunkEmbedding.kb_id == self.kb_id,
        ).delete()
        self.db.commit()

    def search(self, query_vector: Vector, top_k: int = 10) -> list[VectorSearchResult]:
        """
        Searches for the top-k closest vectors to the given query vector.

        Args:
            query_vector: The query vector embedding.
            top_k: The number of results to return.

        Returns:
            A list of dictionaries containing the metadata and similarity scores of
            the top-k results.
        """
        min_score = 0.26
        matching_embeddings = (
            self.db.query(
                ChunkEmbedding,
                ChunkEmbedding.vector.max_inner_product(query_vector),
            )
            .filter(
                # and_ needs to be imported from SQLALchemy
                and_(
                    ChunkEmbedding.kb_id == self.kb_id,
                    ChunkEmbedding.vector.max_inner_product(query_vector)
                    <= -1 * min_score,
                )
            )
            .order_by(ChunkEmbedding.vector.max_inner_product(query_vector).asc())
            .all()
        )

        top_k_matching_embeddings = (
            matching_embeddings[:top_k] if top_k > 0 else matching_embeddings
        )

        search_results: list[VectorSearchResult] = []
        for mp in top_k_matching_embeddings:
            result: VectorSearchResult = VectorSearchResult(
                doc_id=mp[0].transcript_id,
                vector=None,
                metadata=mp[0].chunk_metadata,
                similarity=mp[1],
            )
            search_results.append(result)
        return search_results

    def delete(self) -> None:
        """
        Deletes the vector database.
        """
        self.db.query(ChunkEmbedding).filter(
            ChunkEmbedding.kb_id == self.kb_id
        ).delete()
        self.db.commit()

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "subclass_name": self.__class__.__name__,
            "kb_id": self.kb_id,
            "init_timeout": self.init_timeout,
            "query_timeout": self.query_timeout,
            "insert_timeout": self.insert_timeout,
        }
