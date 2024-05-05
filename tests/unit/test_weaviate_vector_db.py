import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sprag.vector_db_connectors.weaviate_vector_db import WeaviateVectorDB


class TestWeaviateVectorDB(unittest.TestCase):
    def setUp(self):
        self.db = WeaviateVectorDB(
            class_name="TestDocument", kb_id="test_kb", use_embedded_weaviate=True
        )
        return super().setUp()

    def tearDown(self):
        # Consider adding cleanup logic here if necessary (e.g., deleting test data from Weaviate)
        self.db.close()
        return super().tearDown()

    def test_add_vectors_and_search(self):
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata = [
            {"doc_id": "1", "chunk_text": "This is document 1."},
            {"doc_id": "2", "chunk_text": "This is document 2."},
        ]
        self.db.add_vectors(vectors, metadata)

        query_vector = np.array([1, 0])
        results = self.db.search(query_vector, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["doc_id"], "1")
        self.assertGreaterEqual(results[0]["similarity"], 0.99)

    def test_remove_document(self):
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata = [
            {"doc_id": "1", "chunk_text": "This is document 1."},
            {"doc_id": "2", "chunk_text": "This is document 2."},
        ]
        self.db.add_vectors(vectors, metadata)

        self.db.remove_document("1")

        # Verify document removal indirectly (Weaviate doesn't provide a direct way to list documents)
        query_vector = np.array([1, 0])
        results = self.db.search(query_vector, top_k=2)
        self.assertEqual(len(results), 1)  # Expect 1 result as document 1 is deleted
        self.assertEqual(results[0]["metadata"]["doc_id"], "2")


if __name__ == "__main__":
    unittest.main()
