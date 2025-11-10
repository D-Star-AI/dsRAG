import os
import sys
import shutil
import tempfile
import unittest

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from dsrag.knowledge_base import KnowledgeBase
from dsrag.dsparse.file_parsing.vlm_clients import GeminiVLM
from dsrag.reranker import NoReranker


class TestVLMMetaRoundTrip(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="dsrag_kb_test_")
        self.kb_id = "kb_vlm_meta_rt"

    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_kb_persists_vlm_client_in_metadata(self):
        # Create KB with a GeminiVLM client set
        kb = KnowledgeBase(
            kb_id=self.kb_id,
            storage_directory=self.temp_dir,
            vlm_client=GeminiVLM(model="gemini-2.0-flash"),
            reranker=NoReranker(),
        )
        # Force a save and then reload
        kb._save()

        # Re-initialize KB - should load from metadata
        kb2 = KnowledgeBase(
            kb_id=self.kb_id,
            storage_directory=self.temp_dir,
            exists_ok=True,
        )
        self.assertIsNotNone(kb2.vlm_client)
        # Ensure correct type + model preserved
        from dsrag.dsparse.file_parsing.vlm_clients import GeminiVLM as _GeminiVLM
        self.assertIsInstance(kb2.vlm_client, _GeminiVLM)
        self.assertEqual(kb2.vlm_client.model, "gemini-2.0-flash")


if __name__ == "__main__":
    unittest.main()
