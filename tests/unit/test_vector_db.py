from typing import Sequence
import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.database.vector import BasicVectorDB, VectorDB, WeaviateVectorDB, ChromaDB
from dsrag.database.vector.types import ChunkMetadata


class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.storage_directory = "~/test__vector_db_dsRAG"
        self.kb_id = "test_db"
        return super().setUp()

    def tearDown(self):
        storage_path = os.path.join(
            self.storage_directory, "vector_storage", f"{self.kb_id}.pkl"
        )
        if os.path.exists(storage_path):
            os.remove(storage_path)
        return super().tearDown()

    def test__add_vectors_and_search(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        db.add_vectors(vectors, metadata)
        query_vector = np.array([1, 0])
        results = db.search(query_vector, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["doc_id"], "1")
        self.assertGreaterEqual(results[0]["similarity"], 0.99)

    def test__remove_document(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        db.add_vectors(vectors, metadata)
        db.remove_document("1")

        print(db.metadata)
        self.assertEqual(len(db.metadata), 1)
        self.assertEqual(db.metadata[0]["doc_id"], "2")

    def test__empty_search(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        query_vector = np.array([1, 0])
        results = db.search(query_vector)

        self.assertEqual(len(results), 0)

    def test__save_and_load(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        db.add_vectors(vectors, metadata)
        db.save()

        new_db = BasicVectorDB(self.kb_id, self.storage_directory)
        new_db.load()

        self.assertEqual(len(new_db.metadata), 2)
        self.assertEqual(new_db.metadata[0]["doc_id"], "1")
        self.assertEqual(new_db.metadata[1]["doc_id"], "2")

    def test__load_from_dict(self):
        config = {
            "subclass_name": "BasicVectorDB",
            "kb_id": "test_db",
            "storage_directory": "/tmp",
        }
        vector_db_instance = VectorDB.from_dict(config)
        self.assertIsInstance(vector_db_instance, BasicVectorDB)
        self.assertEqual(vector_db_instance.kb_id, "test_db")

    def test__save_and_load_from_dict(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        config = db.to_dict()
        vector_db_instance = VectorDB.from_dict(config)
        self.assertIsInstance(vector_db_instance, BasicVectorDB)
        self.assertEqual(vector_db_instance.kb_id, "test_db")

    def test__assertion_error_on_mismatched_input_lengths(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        vectors = [np.array([1, 0])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        with self.assertRaises(ValueError) as context:
            db.add_vectors(vectors, metadata)
        self.assertTrue(
            "Error in add_vectors: the number of vectors and metadata items must be the same."
            in str(context.exception)
        )

    def test__faiss_search(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory, use_faiss=True)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        db.add_vectors(vectors, metadata)
        query_vector = np.array([1, 0])

        faiss_results = db.search(query_vector, top_k=1)

        db.use_faiss = False
        non_faiss_results = db.search(query_vector, top_k=1)

        self.assertEqual(faiss_results, non_faiss_results)

    def test__delete(self):

        db = BasicVectorDB(self.kb_id, self.storage_directory, use_faiss=True)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        db.add_vectors(vectors, metadata)

        # Make sure the storage directory exists before deleting it
        self.assertTrue(os.path.exists(db.vector_storage_path))
        db.delete()
        # Make sure the storage directory does not exist
        self.assertFalse(os.path.exists(db.vector_storage_path))

    def test__top_k_greater_than_num_vectors(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        db.add_vectors(vectors, metadata)
        query_vector = np.array([1, 0])

        db.use_faiss = True
        results = db.search(query_vector, top_k=3)
        self.assertEqual(len(results), 2)

        db.use_faiss = False
        results = db.search(query_vector, top_k=3)
        self.assertEqual(len(results), 2)


class TestChromaDB(unittest.TestCase):
    def setUp(self):
        self.kb_id = "test_chroma_kb"
        self.db = ChromaDB(kb_id=self.kb_id)
        return super().setUp()

    def tearDown(self):
        # delete test data from ChromaDB
        self.db.delete()
        return super().tearDown()
    
    def test__add_vectors_and_search(self):
        db = ChromaDB(kb_id=self.kb_id)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        db.add_vectors(vectors, metadata)
        query_vector = np.array([[1, 0]])
        results = db.search(query_vector, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["doc_id"], "1")
        self.assertGreaterEqual(results[0]["similarity"], 0.99)

    def test__search_with_metadata_filter(self):
        db = ChromaDB(kb_id=self.kb_id)
        vectors = [np.array([1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 0])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "3",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
            {
                "doc_id": "4",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        db.add_vectors(vectors, metadata)

        query_vector = np.array([[1, 0]])
        metadata_filter = {"field": "doc_id", "operator": "equals", "value": "1"}
        results = db.search(query_vector, top_k=4, metadata_filter=metadata_filter)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["doc_id"], "1")

        # Test with the 'in' operator
        metadata_filter = {"field": "doc_id", "operator": "in", "value": ["1", "4"]}
        results = db.search(query_vector, top_k=4, metadata_filter=metadata_filter)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["metadata"]["doc_id"], "1")
        self.assertEqual(results[1]["metadata"]["doc_id"], "4")

    def test__remove_document(self):
        db = ChromaDB(kb_id=self.kb_id)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        db.add_vectors(vectors, metadata)
        db.remove_document("1")

        num_vectors = db.get_num_vectors()
        self.assertEqual(num_vectors, 1)

    def test__empty_search(self):
        db = ChromaDB(kb_id="test_chroma_db_2")
        query_vector = np.array([1, 0])
        results = db.search(query_vector)

        self.assertEqual(len(results), 0)
        db.delete()

    def test__assertion_error_on_mismatched_input_lengths(self):
        db = ChromaDB(kb_id=self.kb_id)
        vectors = [np.array([1, 0])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

        with self.assertRaises(ValueError) as context:
            db.add_vectors(vectors, metadata)
        self.assertTrue(
            "Error in add_vectors: the number of vectors and metadata items must be the same."
            in str(context.exception)
        )


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

    def test__add_vectors_and_search(self):
        vectors = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "1",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
            {
                "doc_id": "2",
                "chunk_index": 0,
                "chunk_header": "Header3",
                "chunk_text": "Text3",
            },
        ]
        self.db.add_vectors(vectors, metadata)

        query_vector = np.array([1, 0])
        results = self.db.search(query_vector, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["doc_id"], "1")
        self.assertEqual(results[0]["metadata"]["chunk_index"], 0)
        self.assertEqual(results[0]["metadata"]["chunk_header"], "Header1")
        self.assertEqual(results[0]["metadata"]["chunk_text"], "Text1")
        self.assertGreaterEqual(results[0]["similarity"], 0.99)

    def test__remove_document(self):
        vectors = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "1",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
            {
                "doc_id": "2",
                "chunk_index": 0,
                "chunk_header": "Header3",
                "chunk_text": "Text3",
            },
        ]
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
        metadata: Sequence[ChunkMetadata] = [
            {
                "doc_id": "1",
                "chunk_index": 0,
                "chunk_header": "Header1",
                "chunk_text": "Text1",
            },
            {
                "doc_id": "2",
                "chunk_index": 1,
                "chunk_header": "Header2",
                "chunk_text": "Text2",
            },
        ]

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
