from dsrag.database.vector.db import VectorDB
from dsrag.database.vector.types import VectorSearchResult, MetadataFilter
from typing import Optional
import os
import numpy as np
from pinecone import Pinecone
from pinecone import ServerlessSpec


class PineconeDB(VectorDB):
    """
    Class to interact with the Pinecone API. Only supports serverless indexes.
    """

    def __init__(self, kb_id: str, api_key: str, dimension: int = None, cloud: str = "aws", region: str = "us-east-1"):
        """
        Inputs:
            kb_id (str): The name of the KB (which will be used as the name of the index). Note that Pinecone has some restrictions on the name of the index.
            api_key (str): The Pinecone API key. This must always be passed in as an argument, as it does not get saved in the config dictionary.
            dimension (int): The dimension of the vectors. Only required when creating a new index.
            cloud (str): The cloud provider to use. Options are "aws" or "gcp". Only required when creating a new index.
            region (str): The region to use. Only required when creating a new index.
        """
        self.kb_id = kb_id
        self.pc = Pinecone(api_key=api_key)

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
        
        index = self.pc.Index(self.kb_id)

        # create unique ids for each vector
        ids = [f"{meta['doc_id']}_{meta['chunk_index']}" for meta in metadata]
        
        # convert to format that Pinecone expects - list of dictionaries including "values", "id", and "metadata"
        vectors_to_upsert = [{"values": vector, "id": id, "metadata": meta} for vector, id, meta in zip(vectors_as_lists, ids, metadata)]
        
        # upsert the vectors
        index.upsert(vectors_to_upsert)

    def remove_document(self, doc_id: str):
        pass

    def search(self, query_vector, top_k: int = 10, metadata_filter: Optional[MetadataFilter] = None) -> list[VectorSearchResult]:
        index = self.pc.Index(self.kb_id)
        search_results = index.query(vector=query_vector, top_k=top_k)
    
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