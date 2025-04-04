import sys
import os
import unittest
import shutil
from unittest.mock import patch

# add ../../ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dsrag.knowledge_base import KnowledgeBase
from dsrag.llm import OpenAIChatAPI
from dsrag.embedding import CohereEmbedding
from dsrag.database.vector import BasicVectorDB # Import necessary DBs for override
from dsrag.database.chunk import BasicChunkDB
from dsrag.dsparse.file_parsing.file_system import LocalFileSystem

class TestSaveAndLoad(unittest.TestCase):
    def cleanup(self):
        # Use a specific kb_id for testing
        kb = KnowledgeBase(kb_id="test_kb_override", exists_ok=True)
        # Try deleting, ignore if it doesn't exist
        try:
            kb.delete()
        except FileNotFoundError:
            pass
        # Ensure the base KB used in other tests is also cleaned up
        kb_main = KnowledgeBase(kb_id="test_kb", exists_ok=True)
        try:
            kb_main.delete()
        except FileNotFoundError:
            pass

    def test_save_and_load(self):
        self.cleanup()

        # initialize a KnowledgeBase object
        auto_context_model = OpenAIChatAPI(model="gpt-4o-mini")
        embedding_model = CohereEmbedding(model="embed-english-v3.0")   
        kb_id = "test_kb"
        kb = KnowledgeBase(kb_id=kb_id, auto_context_model=auto_context_model, embedding_model=embedding_model, exists_ok=False)

        # load the KnowledgeBase object
        kb1 = KnowledgeBase(kb_id=kb_id)

        # verify that the KnowledgeBase object has the right parameters
        self.assertEqual(kb1.auto_context_model.model, "gpt-4o-mini")
        self.assertEqual(kb1.embedding_model.model, "embed-english-v3.0")

        # delete the KnowledgeBase object
        kb1.delete()

        # Check if the underlying storage path exists after deletion
        chunk_db_path = os.path.join(os.path.expanduser("~/dsRAG"), "chunk_data", kb_id)
        self.assertFalse(os.path.exists(chunk_db_path), f"Chunk DB path {chunk_db_path} should not exist after delete.")

    def test_load_override_warning(self):
        """Test that warnings are logged when overriding components during load."""
        self.cleanup()
        kb_id = "test_kb_override"

        # 1. Create initial KB to save config
        kb_initial = KnowledgeBase(kb_id=kb_id, exists_ok=False)

        # Mock DB/FS instances for overriding
        mock_vector_db = BasicVectorDB(kb_id=kb_id, storage_directory="~/dsRAG_temp_v")
        mock_chunk_db = BasicChunkDB(kb_id=kb_id, storage_directory="~/dsRAG_temp_c")
        mock_file_system = LocalFileSystem(base_path="~/dsRAG_temp_f")

        # 2. Patch logging.warning
        with patch('logging.warning') as mock_warning:
            # 3. Load KB again with overrides
            kb_loaded = KnowledgeBase(
                kb_id=kb_id,
                vector_db=mock_vector_db,
                chunk_db=mock_chunk_db,
                file_system=mock_file_system,
                exists_ok=True # This ensures _load is called
            )

            # 4. Assertions
            self.assertEqual(mock_warning.call_count, 3, "Expected 3 warning calls (vector_db, chunk_db, file_system)")

            expected_calls = [
                f"Overriding stored vector_db for KB '{kb_id}' during load.",
                f"Overriding stored chunk_db for KB '{kb_id}' during load.",
                f"Overriding stored file_system for KB '{kb_id}' during load.",
            ]
            
            call_args_list = [call.args[0] for call in mock_warning.call_args_list]
            call_kwargs_list = [call.kwargs for call in mock_warning.call_args_list]

            # Check if all expected messages are present in the actual calls
            for expected_msg in expected_calls:
                 self.assertTrue(any(expected_msg in str(arg) for arg in call_args_list), f"Expected warning message '{expected_msg}' not found.")

            # Check the 'extra' parameter in all calls
            expected_extra = {'kb_id': kb_id}
            for kwargs in call_kwargs_list:
                self.assertIn('extra', kwargs, "Warning call missing 'extra' parameter.")
                self.assertEqual(kwargs['extra'], expected_extra, f"Warning call 'extra' parameter mismatch. Expected {expected_extra}, got {kwargs['extra']}")


        # 5. Cleanup
        kb_loaded.delete()
        # Explicitly try to remove temp dirs if they exist (though kb.delete should handle them)
        temp_dirs = ["~/dsRAG_temp_v", "~/dsRAG_temp_c", "~/dsRAG_temp_f"]
        for temp_dir in temp_dirs:
            expanded_path = os.path.expanduser(temp_dir)
            if os.path.exists(expanded_path):
                try:
                    shutil.rmtree(expanded_path) # Use shutil.rmtree
                    # print(f"Cleaned up temp dir: {expanded_path}") # Optional: uncomment for debug
                except OSError as e:
                    print(f"Error removing temp dir {expanded_path}: {e}") # Log error if removal fails


if __name__ == "__main__":
    unittest.main()