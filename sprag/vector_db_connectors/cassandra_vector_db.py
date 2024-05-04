import cassio
from cassio.table import MetadataVectorCassandraTable
from sprag.vector_db import VectorDB


class CassandraVectorDB(VectorDB):

    def __init__(
        self,
        contact_points,
        keyspace,
        table,
        username=None,
        password=None,
    ):
        """
        Initializes a CassandraVectorDB instance.

        Args:
            contact_points (list): List of contact points for the Cassandra cluster.
            keyspace (str): Name of the keyspace to use.
            table (str): Name of the table to use.
            username (str, optional): Username for authentication. Defaults to None.
            password (str, optional): Password for authentication. Defaults to None.
        """
        self.table = table
        self.v_table = None

        cassio.init(
            contact_points=contact_points,
            keyspace=keyspace,
            username=username,
            password=password,
        )
        cassio.config.resolve_session().execute(
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )

    def add_vectors(self, vectors, metadata):
        """
        Adds vectors and their associated metadata to Cassandra.

        Args:
            vectors (list): A list of vectors (numpy arrays) to add.
            metadata (list): A list of dictionaries containing metadata for each vector.
        """
        if self.v_table is None and vectors is not None and len(vectors) > 0:
            self.v_table = MetadataVectorCassandraTable(
                table=self.table,
                vector_dimension=len(vectors[0]),
                primary_key_type="TEXT",
            )

        for vector, meta in zip(vectors, metadata):
            vector_list = vector.tolist()
            self.v_table.put_async(
                row_id=meta["doc_id"],  # Assuming doc_id is used as the primary key
                vector=vector_list,
                metadata=meta,
            ).result()

    def search(self, query_vector, top_k=10):
        """
        Searches for the top-k most similar vectors to the query vector using cosine similarity.

        Args:
            query_vector (np.array): The query vector.
            top_k (int, optional): The number of results to return. Defaults to 10.

        Returns:
            list: A list of dictionaries containing the metadata and similarity scores of the results.
        """
        if self.v_table is None:
            return []

        results = self.v_table.metric_ann_search(
            vector=query_vector.tolist(), n=top_k, metric="cos"
        )

        return [
            {
                "metadata": match["metadata"],
                "similarity": match[
                    "distance"
                ],  # Distance is "similarity" in this version of Cassio
            }
            for match in results
        ]

    def remove_document(self, doc_id):
        """
        Removes a document (vector and metadata) from Cassandra based on the document ID.

        Args:
            doc_id (str): The ID of the document to remove.
        """
        self.v_table.delete(row_id=doc_id)
