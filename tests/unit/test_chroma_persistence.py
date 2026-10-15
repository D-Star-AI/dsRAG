import importlib.util
import subprocess
import sys
import tempfile
import textwrap
import unittest


@unittest.skipUnless(importlib.util.find_spec("chromadb"), "ChromaDB is not installed")
class TestChromaPersistence(unittest.TestCase):
    def test_custom_storage_survives_kb_reload(self):
        # Chroma retains file handles for the life of the process on Windows.
        # End the client process before cleaning up its temporary storage.
        script = textwrap.dedent('''
            import sys
            from dsrag.database.vector import ChromaDB, VectorDB
            from dsrag.embedding import Embedding
            from dsrag.knowledge_base import KnowledgeBase
            from dsrag.llm import LLM
            from dsrag.reranker import NoReranker

            class PersistenceEmbedding(Embedding):
                def __init__(self, dimension=2):
                    super().__init__(dimension)

                def get_embeddings(self, text, input_type=None):
                    return [[1.0, 0.0] for _ in text]

            class PersistenceLLM(LLM):
                def make_llm_call(self, chat_messages):
                    raise AssertionError("AutoContext generation is disabled")

            directory = sys.argv[1]
            db = ChromaDB("chroma-persistence-test", directory)
            kb = KnowledgeBase(
                "chroma-persistence-test", storage_directory=directory,
                embedding_model=PersistenceEmbedding(), reranker=NoReranker(),
                auto_context_model=PersistenceLLM(), vector_db=db,
            )
            kb.add_document(
                "manual", text="The orange sensor is calibrated weekly.",
                semantic_sectioning_config={"use_semantic_sectioning": False},
                auto_context_config={"use_generated_title": False, "get_document_summary": False},
            )
            initial = kb.query(["orange sensor"])
            assert initial[0]["doc_id"] == "manual"
            loaded = KnowledgeBase("chroma-persistence-test", storage_directory=directory)
            assert loaded.vector_db.storage_directory == directory
            assert loaded.query(["orange sensor"]) == initial
            rebuilt = VectorDB.from_dict(db.to_dict())
            assert rebuilt.storage_directory == directory
            loaded.delete()
        ''')
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", script, directory],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
