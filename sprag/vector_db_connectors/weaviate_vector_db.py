import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from sprag.vector_db import VectorDB


class WeaviateVectorDB(VectorDB):
    """
    An implementation of the VectorDB interface for Weaviate using the Python v4 client.

    This class provides methods for adding, removing, and searching for vectorized data
    within a Weaviate instance.
    """

    def __init__(
        self,
        http_host="localhost",
        http_port="8099",
        http_secure=False,
        grpc_host="localhost",
        grpc_port="50052",
        grpc_secure=False,
        weaviate_secret="secr3tk3y",
        openai_api_key: str = None,
        class_name="Document",
        kb_id: str = "WeaviateVectorDBKnowledgeBase",
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
        additional_headers = {}
        if openai_api_key != None:
            additional_headers = {"X-OpenAI-Api-Key": openai_api_key}

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
        self.class_name = class_name
        self.kb_id = kb_id

        self.collection = self.client.collections.get(kb_id)

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
                uuid = generate_uuid5(doc_id)
                batch.add_object(
                    properties={
                        "content": chunk_text,
                        "doc_id": doc_id,
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
        uuid = generate_uuid5(doc_id)
        self.collection.data.delete_by_id(uuid)

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
        results = []
        response = self.collection.query.near_vector(
            near_vector=query_vector.tolist(),
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
