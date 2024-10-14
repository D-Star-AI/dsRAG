import os
from typing import Optional, Sequence

from dsrag.database.vector import VectorSearchResult
from dsrag.database.vector.db import VectorDB

from dsrag.database.vector.types import MetadataFilter, Vector, ChunkMetadata


def _convert_metadata_to_expr(metadata_filter: MetadataFilter) -> str:
    """
    Convert the metadata filter to the expression used in the Milvus query method.

    Args:
        metadata_filter (dict): The metadata filter.

    Returns:
        str: The formatted expression.
    """

    if not metadata_filter:
        return ""
    field = metadata_filter["field"]
    operator = metadata_filter["operator"]
    value = metadata_filter["value"]

    operator_mapping = {
        "equals": "==",
        "not_equals": "!=",
        "in": "in",
        "not_in": "not in",
        "greater_than": ">",
        "less_than": "<",
        "greater_than_equals": ">=",
        "less_than_equals": "<=",
    }

    formatted_operator = operator_mapping[operator]
    if isinstance(value, str):
        value = f'"{value}"'
    formatted_metadata_filter = f"metadata['{field}'] {formatted_operator} {value}"

    return formatted_metadata_filter


class MilvusDB(VectorDB):
    def __init__(self, kb_id: str, storage_directory: str = '~/dsRAG',
                 dimension: int = 768):
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError(
                "MilvusDB requires the pymilvus library. "
                "Please install it by running `pip install pymilvus`."
            )
        self.kb_id = kb_id
        # Expand user path and ensure directory exists
        self.storage_directory = os.path.expanduser(storage_directory)
        self.vector_storage_path = os.path.join(self.storage_directory,
                                                'vector_storage')
        os.makedirs(self.vector_storage_path,
                    exist_ok=True)  # Create directory if it doesn't exist

        # Initialize Milvus client
        self.client = MilvusClient(uri=f"{self.vector_storage_path}.db")
        self._create_collection(collection_name=self.kb_id, dimension=dimension)

    def _create_collection(self, collection_name: str, dimension: int = 768):
        from pymilvus import DataType
        if self.client.has_collection(collection_name=collection_name):
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR,
                         is_primary=True, max_length=100)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR,
                         dim=dimension)
        schema.add_field(field_name="metadata", datatype=DataType.JSON,
                         enable_dynamic_field=True)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True
        )

    def add_vectors(self, vectors: Sequence[Vector], metadata: Sequence[ChunkMetadata]):
        try:
            assert len(vectors) == len(metadata)
        except AssertionError:
            raise ValueError(
                'Error in add_vectors: the number of vectors and metadata items must be the same.')

        # Create the ids from the metadata, defined as {metadata["doc_id"]}_{metadata["chunk_index"]}
        ids = [f"{meta['doc_id']}_{meta['chunk_index']}" for meta in metadata]

        data = []
        for i, vector in enumerate(vectors):
            data.append({
                'doc_id': ids[i],
                'vector': vector,
                'metadata': metadata[i]
            })

        self.client.upsert(
            collection_name=self.kb_id,
            data=data
        )

    def search(self, query_vector, top_k: int=10, metadata_filter: Optional[dict] = None) -> list[VectorSearchResult]:
        query_results = self.client.search(
            collection_name=self.kb_id,
            data=[query_vector],
            filter=_convert_metadata_to_expr(metadata_filter),
            limit=top_k,
            output_fields=["*"]
        )[0]
        results: list[VectorSearchResult] = []
        for res in query_results:
            results.append(
                VectorSearchResult(
                    doc_id=res['entity']['doc_id'],
                    metadata=res['entity']['metadata'],
                    similarity=res['distance'],
                    vector=res['entity']['vector'],
                )
            )

        results = sorted(results, key=lambda x: x['similarity'], reverse=True)

        return results

    def remove_document(self, doc_id):
        self.client.delete(
            collection_name=self.kb_id,
            filter=f'metadata["doc_id"] == "{doc_id}"'
        )

    def get_num_vectors(self):
        res = self.client.query(
            collection_name=self.kb_id,
            output_fields=["count(*)"]
        )
        return res[0]["count(*)"]

    def delete(self):
        self.client.drop_collection(collection_name=self.kb_id)

    def to_dict(self):
        return {
            **super().to_dict(),
            'kb_id': self.kb_id,
        }
