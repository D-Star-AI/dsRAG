from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import numpy as np
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
import chromadb


class VectorDB(ABC):
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
    def add_vectors(self, vector, metadata):
        """
        Store a list of vectors with associated metadata.
        """
        pass

    @abstractmethod
    def remove_document(self, doc_id):
        """
        Remove all vectors and metadata associated with a given document ID.
        """
        pass

    @abstractmethod
    def search(self, query_vector, top_k=10):
        """
        Retrieve the top-k closest vectors to a given query vector.
        - needs to return results as list of dictionaries in this format: 
        {
            'metadata': 
                {
                    'doc_id': doc_id, 
                    'chunk_index': chunk_index, 
                    'chunk_header': chunk_header,
                    'chunk_text': chunk_text
                }, 
            'similarity': similarity, 
        }
        """
        pass

    @abstractmethod
    def delete(self):
        """
        Delete the vector database.
        """
        pass


class BasicVectorDB(VectorDB):
    def __init__(self, kb_id: str, storage_directory: str = '~/dsRAG', use_faiss: bool = True):
        self.kb_id = kb_id
        self.storage_directory = storage_directory
        self.use_faiss = use_faiss
        self.vector_storage_path = os.path.join(self.storage_directory, 'vector_storage', f'{kb_id}.pkl')
        self.load()

    def add_vectors(self, vectors, metadata):
        try:
            assert len(vectors) == len(metadata)
        except AssertionError:
            raise ValueError('Error in add_vectors: the number of vectors and metadata items must be the same.')
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)
        self.save()

    def search(self, query_vector, top_k=10):
        if not self.vectors:
            return []
        
        if self.use_faiss:
            return self.search_faiss(query_vector, top_k)

        similarities = cosine_similarity([query_vector], self.vectors)[0]
        indexed_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        results = []
        for i, similarity in indexed_similarities[:top_k]:
            result = {
                'metadata': self.metadata[i],
                'similarity': similarity,
            }
            results.append(result)
        return results
    
    def search_faiss(self, query_vector, top_k=10):
        from faiss.contrib.exhaustive_search import knn
        import numpy as np

        # Limit top_k to the number of vectors we have - Faiss doesn't automatically handle this
        top_k = min(top_k, len(self.vectors))

        # faiss expects 2D arrays of vectors
        vectors_array = np.array(self.vectors).astype('float32').reshape(len(self.vectors), -1)
        query_vector_array = np.array(query_vector).astype('float32').reshape(1, -1)
        
        _, I = knn(query_vector_array, vectors_array, top_k) # I is a list of indices in the corpus_vectors array
        results = []
        for i in I[0]:
            result = {
                'metadata': self.metadata[i],
                'similarity': cosine_similarity([query_vector], [self.vectors[i]])[0][0],
            }
            results.append(result)
        return results

    def remove_document(self, doc_id):
        i = 0
        while i < len(self.metadata):
            if self.metadata[i]['doc_id'] == doc_id:
                del self.vectors[i]
                del self.metadata[i]
            else:
                i += 1
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.vector_storage_path), exist_ok=True)  # Ensure the directory exists
        with open(self.vector_storage_path, 'wb') as f:
            pickle.dump((self.vectors, self.metadata), f)

    def load(self):
        if os.path.exists(self.vector_storage_path):
            with open(self.vector_storage_path, 'rb') as f:
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
            'kb_id': self.kb_id,
            'storage_directory': self.storage_directory,
            'use_faiss': self.use_faiss,
        }


class WeaviateVectorDB(VectorDB):
    """
    An implementation of the VectorDB interface for Weaviate using the Python v4 client.

    This class provides methods for adding, removing, and searching for vectorized data
    within a Weaviate instance.
    """

    def __init__(
        self,
        kb_id: str,
        http_host="localhost",
        http_port="8099",
        http_secure=False,
        grpc_host="localhost",
        grpc_port="50052",
        grpc_secure=False,
        weaviate_secret="secr3tk3y",
        init_timeout: int = 2,
        query_timeout: int = 45,
        insert_timeout: int = 120,
        use_embedded_weaviate: bool = False,
    ):
        """
        Initializes a WeaviateVectorDB instance.

        Args:
            http_host: The hostname of the Weaviate server.
            http_port: The HTTP port of the Weaviate server.
            http_secure: Whether to use HTTPS for the connection.
            grpc_host: The hostname of the Weaviate server for gRPC connections.
            grpc_port: The gRPC port of the Weaviate server.
            grpc_secure: Whether to use gRPCs for the connection.
            class_name: The name of the Weaviate class to use for storing data.
            kb_id: An optional identifier for the knowledge base.
        """
        
        # save all of these parameters as attributes so they're easily accessible for the to_dict method
        self.kb_id = kb_id
        self.http_host = http_host
        self.http_port = http_port
        self.http_secure = http_secure
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.grpc_secure = grpc_secure
        self.weaviate_secret = weaviate_secret
        self.init_timeout = init_timeout
        self.query_timeout = query_timeout
        self.insert_timeout = insert_timeout
        self.use_embedded_weaviate = use_embedded_weaviate

        additional_headers = {}
        if use_embedded_weaviate:
            additional_headers["ENABLE_MODULES"] = (
                "backup-filesystem,text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai"
            )
            additional_headers["BACKUP_FILESYSTEM_PATH"] = "./weaviate/backups"
            self.client = weaviate.WeaviateClient(
                embedded_options=weaviate.embedded.EmbeddedOptions(
                    persistence_data_path="./weaviate/data",
                ),
                additional_headers=additional_headers,
            )
        else:
            connection_params = weaviate.connect.ConnectionParams.from_params(
                http_host=http_host,
                http_port=http_port,
                http_secure=http_secure,
                grpc_host=grpc_host,
                grpc_port=grpc_port,
                grpc_secure=grpc_secure,
            )
            self.client = weaviate.WeaviateClient(
                connection_params=connection_params,
                auth_client_secret=weaviate.auth.AuthApiKey(weaviate_secret),
                additional_headers=additional_headers,
                additional_config=weaviate.classes.init.AdditionalConfig(
                    timeout=weaviate.classes.init.Timeout(
                        init=init_timeout, query=query_timeout, insert=insert_timeout
                    )
                ),
            )
        
        self.client.connect()
        self.collection = self.client.collections.get(kb_id) # assume this creates a new collection if it doesn't exist

    def close(self):
        """
        Closes the connection to Weaviate.
        """
        self.client.close()

    def add_vectors(self, vectors, metadata):
        """
        Adds a list of vectors with associated metadata to Weaviate.

        Args:
            vectors: A list of vector embeddings.
            metadata: A list of dictionaries containing metadata for each vector.

        Raises:
            ValueError: If the number of vectors and metadata items do not match.
        """
        try:
            assert len(vectors) == len(metadata)
        except AssertionError as exc:
            raise ValueError(
                "Error in add_vectors: the number of vectors and metadata items must be the same."
            ) from exc

        with self.collection.batch.dynamic() as batch:
            for vector, meta in zip(vectors, metadata):
                doc_id = meta.get("doc_id", "")
                chunk_text = meta.get("chunk_text", "")
                chunk_index = meta.get("chunk_index", 0)
                uuid = generate_uuid5(f"{doc_id}_{chunk_index}")
                batch.add_object(
                    properties={
                        "content": chunk_text,
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "metadata": meta,
                    },
                    vector=vector,
                    uuid=uuid,
                )

    def remove_document(self, doc_id):
        """
        Removes a document (data object) from Weaviate.

        Args:
            doc_id: The UUID of the document to remove.
        """
        self.collection.data.delete_many(
            where=wvc.query.Filter.by_property("doc_id").contains_any([doc_id])
        )

    def search(self, query_vector, top_k=10):
        """
        Searches for the top-k closest vectors to the given query vector.

        Args:
            query_vector: The query vector embedding.
            top_k: The number of results to return.

        Returns:
            A list of dictionaries containing the metadata and similarity scores of
            the top-k results.
        """
        # convert the query vector to a list if it's not already
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        results = []
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )
        for obj in response.objects:
            results.append(
                {
                    "doc_id": obj.properties["doc_id"],
                    "metadata": obj.properties["metadata"],
                    "similarity": 1.0 - obj.metadata.distance,
                    "vector": obj.vector,
                }
            )
        return results

    def to_dict(self):
        return {
            **super().to_dict(),
            'kb_id': self.kb_id,
            'http_host': self.http_host,
            'http_port': self.http_port,
            'http_secure': self.http_secure,
            'grpc_host': self.grpc_host,
            'grpc_port': self.grpc_port,
            'grpc_secure': self.grpc_secure,
            'weaviate_secret': self.weaviate_secret,
            'init_timeout': self.init_timeout,
            'query_timeout': self.query_timeout,
            'insert_timeout': self.insert_timeout,
            'use_embedded_weaviate': self.use_embedded_weaviate,
        }
    
    def delete(self):
        pass


class ChromaDB(VectorDB):

    def __init__(self, kb_id: str, storage_directory: str = '~/dsRAG'):
        self.kb_id = kb_id
        self.storage_directory = os.path.expanduser(storage_directory)
        self.vector_storage_path = os.path.join(self.storage_directory, 'vector_storage')
        self.client = chromadb.PersistentClient(path=self.vector_storage_path)
        self.collection = self.client.get_or_create_collection(kb_id, metadata={"hnsw:space": "cosine"})
    
    def get_num_vectors(self):
        return self.collection.count()
    
    def add_vectors(self, vectors, metadata):
        try:
            assert len(vectors) == len(metadata)
        except AssertionError:
            raise ValueError('Error in add_vectors: the number of vectors and metadata items must be the same.')
        
        # Create the ids from the metadata, defined as {metadata["doc_id"]}_{metadata["chunk_index"]}
        ids = [f"{meta['doc_id']}_{meta['chunk_index']}" for meta in metadata]

        self.collection.add(
            embeddings=vectors,
            metadatas=metadata,
            ids=ids
        )
    
    def search(self, query_vector, top_k=10):

        num_vectors = self.get_num_vectors()
        if num_vectors == 0:
            return []
            #raise ValueError('No vectors stored in the database.')
        
        query_results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["distances", "metadatas"]
        )
        
        metadata = query_results["metadatas"][0]
        distances = query_results["distances"][0]

        results = []
        for _, (distance, metadata) in enumerate(zip(distances, metadata)):
            results.append({
                'metadata': metadata,
                'similarity': 1 - distance,
            })
        
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)

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
            'kb_id': self.kb_id,
        }