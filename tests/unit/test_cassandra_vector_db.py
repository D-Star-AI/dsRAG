import sys
import os
import unittest
import cassio
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sprag.vector_db_connectors.cassandra_vector_db import CassandraVectorDB

CONTACT_POINTS = ["cassandra", "127.0.0.1"]
KEYSPACE = "my_keyspace"
TABLE = "my_vector_table"
USERNAME = None
PASSWORD = None


class TestCassandraVectorDB(unittest.TestCase):
    def setUp(self):
        self.db = CassandraVectorDB(CONTACT_POINTS, KEYSPACE, TABLE, USERNAME, PASSWORD)
        # Clear the table before each test
        my_table = cassio.table.MetadataVectorCassandraTable(
            table=TABLE,
            vector_dimension=len(np.array([1, 2, 3])),
            primary_key_type="TEXT",
        )
        my_table.clear()

    def test_add_and_search_vectors(self):
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        metadata = [
            {
                "doc_id": "doc1",
                "chunk_index": 1,
                "chunk_header": "Header 1",
                "chunk_text": "Text 1",
            },
            {
                "doc_id": "doc2",
                "chunk_index": 2,
                "chunk_header": "Header 2",
                "chunk_text": "Text 2",
            },
            {
                "doc_id": "doc3",
                "chunk_index": 3,
                "chunk_header": "Header 3",
                "chunk_text": "Text 3",
            },
        ]
        self.db.add_vectors(vectors, metadata)

        query_vector = np.array([1, 2, 3])
        results = self.db.search(query_vector, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["metadata"]["doc_id"], "doc1")
        self.assertEqual(results[0]["doc_id"], "doc1")
        self.assertAlmostEqual(results[0]["similarity"], 1.0)

    def test_remove_document(self):
        vectors = [np.array([1, 2, 3])]
        metadata = [
            {
                "doc_id": "doc1",
                "chunk_index": 1,
                "chunk_header": "Header 1",
                "chunk_text": "Text 1",
            }
        ]
        self.db.add_vectors(vectors, metadata)

        self.db.remove_document("doc1")
        results = self.db.search(np.array([1, 2, 3]))
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
