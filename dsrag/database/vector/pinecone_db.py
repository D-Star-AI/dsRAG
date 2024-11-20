from dsrag.database.vector.db import VectorDB
from dsrag.database.vector.types import VectorSearchResult, MetadataFilter
from typing import Optional
import os
import numpy as np
from pinecone import Pinecone
from pinecone import ServerlessSpec


def format_metadata_filter(metadata_filter: MetadataFilter) -> dict:

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
    formatted_operator = operator_mapping.get(operator)

    if formatted_operator == "$eq":
        formatted_metadata_filter = {field: value}
    else:
        formatted_metadata_filter = {field: {formatted_operator: value}}
    
    return formatted_metadata_filter



class PineconeDB(VectorDB):
    """
    Class to interact with the Pinecone API. Only supports serverless indexes.
    """

    def __init__(self, kb_id: str, dimension: int = None, cloud: str = "aws", region: str = "us-east-1"):
        """
        Inputs:
            kb_id (str): The name of the KB (which will be used as the name of the index). Note that Pinecone has some restrictions on the name of the index.
            api_key (str): The Pinecone API key. This must always be passed in as an argument, as it does not get saved in the config dictionary.
            dimension (int): The dimension of the vectors. Only required when creating a new index.
            cloud (str): The cloud provider to use. Options are "aws" or "gcp". Only required when creating a new index.
            region (str): The region to use. Only required when creating a new index.
        """

        # Format the kb_id in case it contains any invalid characters
        # There can't be any underscores in the kb_id
        kb_id = kb_id.replace("_", "-")
        kb_id = kb_id.replace(" ", "-")

        self.kb_id = kb_id
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        # See if the index already exists
        existing_indexes = self.pc.list_indexes()
        existing_index_names = [index["name"] for index in existing_indexes]
        if kb_id not in existing_index_names:
            if dimension is None:
                raise ValueError("Dimension must be specified when creating a new index.")
            self.pc.create_index(name=kb_id, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud=cloud, region=region))

    def add_vectors(self, vectors: list, metadata: list):
        # Convert NumPy arrays to lists
        vectors_as_lists = [vector.tolist() if isinstance(vector, np.ndarray) else vector for vector in vectors]
        try:
            assert len(vectors_as_lists) == len(metadata)
        except AssertionError:
            raise ValueError(
                "Error in add_vectors: the number of vectors and metadata items must be the same."
            )
        
        # Make sure each value in the vectors is a float and not an int
        for vector in vectors_as_lists:
            for i, value in enumerate(vector):
                if isinstance(value, int):
                    vector[i] = float(value)
        
        index = self.pc.Index(self.kb_id)

        # create unique ids for each vector
        ids = [f"{meta['doc_id']}_{meta['chunk_index']}" for meta in metadata]
        
        # convert to format that Pinecone expects - list of dictionaries including "values", "id", and "metadata"
        vectors_to_upsert = [{"values": vector, "id": id, "metadata": meta} for vector, id, meta in zip(vectors_as_lists, ids, metadata)]
        
        # upsert the vectors
        index.upsert(vectors_to_upsert)

    def get_num_vectors(self):
        index = self.pc.Index(self.kb_id)
        stats = index.describe_index_stats()
        if stats:
            num_vectors = stats["total_vector_count"]
            return num_vectors
        else:
            return 0

    def remove_document(self, doc_id: str):
        index = self.pc.Index(self.kb_id)
        # The doc id is in the metadata
        for ids in index.list(prefix=doc_id):
            index.delete(ids=ids)

    def search(self, query_vector, top_k: int = 10, metadata_filter: Optional[MetadataFilter] = None) -> list[VectorSearchResult]:
        index = self.pc.Index(self.kb_id)
        # Convert the query vector to a list if it is a NumPy array
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        if metadata_filter:
            formatted_metadata_filter = format_metadata_filter(metadata_filter)
            search_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True, filter=formatted_metadata_filter)
        else:
            search_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

        results = []
        for match in search_results['matches']:
            doc_id = match['metadata'].get('doc_id') or match.get('id')
            similarity = match['score']

            results.append(
                VectorSearchResult(
                    doc_id=doc_id,
                    vector=None,
                    metadata=match['metadata'],
                    similarity=similarity
                )
            )

        results = sorted(results, key=lambda x: x["similarity"], reverse=True)

        return results
    
    def delete(self):
        """
        WARNING: This will permanently delete the index and all associated data.
        """
        self.pc.delete_index(self.kb_id)

    def to_dict(self):
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
        }