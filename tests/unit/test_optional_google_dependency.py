import subprocess
import sys
import textwrap
import unittest


class TestOptionalGoogleDependency(unittest.TestCase):
    def test_knowledge_base_without_google_generativeai(self):
        # Use a fresh interpreter so an installed/cached Google SDK cannot hide
        # an eager import. Custom components are a supported KB configuration.
        script = textwrap.dedent('''
            import importlib.abc
            import importlib.util
            import sys
            import tempfile

            class NoGoogleSDK(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "google.generativeai" or fullname.startswith("google.generativeai."):
                        raise ModuleNotFoundError("Google SDK is not installed", name=fullname)

            sys.meta_path.insert(0, NoGoogleSDK())
            # Optional integrations may probe availability before importing.
            original_find_spec = importlib.util.find_spec
            def find_spec(name, package=None):
                if name == "google.generativeai":
                    return None
                return original_find_spec(name, package)
            importlib.util.find_spec = find_spec
            from dsrag.knowledge_base import KnowledgeBase
            from dsrag.embedding import Embedding
            from dsrag.llm import LLM
            from dsrag.reranker import NoReranker

            class FixedEmbedding(Embedding):
                def __init__(self, dimension=2):
                    super().__init__(dimension)

                def get_embeddings(self, text, input_type=None):
                    return [[1.0, 0.0] for _ in text]

            class UnusedLLM(LLM):
                def make_llm_call(self, chat_messages):
                    raise AssertionError("AutoContext generation is disabled")

            with tempfile.TemporaryDirectory() as directory:
                kb = KnowledgeBase(
                    "optional-google-test", storage_directory=directory,
                    embedding_model=FixedEmbedding(), reranker=NoReranker(),
                    auto_context_model=UnusedLLM(),
                )
                kb.add_document(
                    "manual", text="The orange sensor is calibrated weekly.",
                    semantic_sectioning_config={"use_semantic_sectioning": False},
                    auto_context_config={"use_generated_title": False, "get_document_summary": False},
                )
                results = kb.query(["orange sensor"])
                assert results[0]["doc_id"] == "manual"
                assert "orange sensor" in results[0]["content"]
                kb.delete()
            assert "google.generativeai" not in sys.modules
        ''')
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
