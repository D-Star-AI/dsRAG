import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.vector_db_connectors.weaviate_vector_db import WeaviateVectorDB
from dsrag.vector_db import VectorDB

class TestWeaviateVectorDB(unittest.TestCase):
    def setUp(self):
        self.kb_id = "test_kb"
        self.db = WeaviateVectorDB(kb_id=self.kb_id, use_embedded_weaviate=True)
        return super().setUp()

    def tearDown(self):
        # delete test data from Weaviate
        self.db.client.collections.delete(self.kb_id)
        self.db.close()
        return super().tearDown()

    def test_add_vectors_and_search(self):
        vectors = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
        metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                    {'doc_id': '1', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'},
                    {'doc_id': '2', 'chunk_index': 0, 'chunk_header': 'Header3', 'chunk_text': 'Text3'}]
        self.db.add_vectors(vectors, metadata)

        query_vector = np.array([1, 0])
        results = self.db.search(query_vector, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["doc_id"], "1")
        self.assertEqual(results[0]["metadata"]["chunk_index"], 0)
        self.assertEqual(results[0]["metadata"]["chunk_header"], "Header1")
        self.assertEqual(results[0]["metadata"]["chunk_text"], "Text1")
        self.assertGreaterEqual(results[0]["similarity"], 0.99)

    def test_remove_document(self):
        vectors = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
        metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                    {'doc_id': '1', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'},
                    {'doc_id': '2', 'chunk_index': 0, 'chunk_header': 'Header3', 'chunk_text': 'Text3'}]
        self.db.add_vectors(vectors, metadata)

        self.db.remove_document("1")

        # Verify document removal indirectly (Weaviate doesn't provide a direct way to list documents)
        query_vector = np.array([1, 0])
        results = self.db.search(query_vector, top_k=3)
        # Expect 1 result as document 1 is deleted and document 2 only has 1 chunk
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["doc_id"], "2")

    def test__save_and_load(self):
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                    {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
        
        self.db.add_vectors(vectors, metadata)
        self.db.close()
        
        # load the saved db
        self.db = WeaviateVectorDB(kb_id=self.kb_id, use_embedded_weaviate=True)
        
        # Verify data existence indirectly (Weaviate doesn't provide a direct way to list documents)
        query_vector = np.array([1, 0])
        results = self.db.search(query_vector, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["metadata"]["doc_id"], "1")

    def test__save_and_load_from_dict(self):
        config = self.db.to_dict()
        self.db.close()

        self.db = VectorDB.from_dict(config)
        self.assertIsInstance(self.db, WeaviateVectorDB)
        self.assertEqual(self.db.kb_id, self.kb_id)


if __name__ == "__main__":
    unittest.main()