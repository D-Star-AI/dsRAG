import os
import sys
import shutil
import tempfile
import unittest
from glob import glob

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from dsrag.knowledge_base import KnowledgeBase
from dsrag.dsparse.file_parsing.file_system import LocalFileSystem
from dsrag.dsparse.file_parsing.vlm_clients import GeminiVLM
from dsrag.reranker import NoReranker
from dsrag.embedding import Embedding
import numpy as np


class DummyEmbedding(Embedding):
    """Minimal embedding implementation for tests (no network)."""
    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def get_embeddings(self, text: list[str], input_type: str = ""):
        # Return deterministic zero vectors of the configured dimension
        return [np.zeros(self.dimension, dtype=np.float32) for _ in text]

    def to_dict(self):
        # Provide a minimal serializable dict (not used for from_dict in this test)
        return {"subclass_name": "DummyEmbedding", "dimension": self.dimension}


class TestAddDocumentWithSerializedVLM(unittest.TestCase):
    @unittest.skipIf('GEMINI_API_KEY' not in os.environ, "GEMINI_API_KEY not found in environment")
    def test_add_document_with_serialized_vlm_and_images_already_exist(self):
        temp_dir = tempfile.mkdtemp(prefix="dsrag_kb_add_doc_")
        try:
            kb_id = "kb_serialized_vlm"
            doc_id = "doc_serialized_vlm"

            # Set up file system and pre-create images under kb/doc
            fs = LocalFileSystem(base_path=os.path.join(temp_dir, "page_images"))
            fs.create_directory(kb_id, doc_id)

            # Copy an existing test image into the kb/doc as page_1.jpg
            src_img = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tests/data/page_7.jpg"))
            dst_img = os.path.join(temp_dir, "page_images", kb_id, doc_id, "page_1.jpg")
            shutil.copyfile(src_img, dst_img)

            # Build KB with dummy embedding and no reranker network
            kb = KnowledgeBase(
                kb_id=kb_id,
                storage_directory=temp_dir,
                embedding_model=DummyEmbedding(dimension=8),
                reranker=NoReranker(),
                file_system=fs,
            )

            # Serialized VLM client for per-document override
            serialized_vlm = GeminiVLM(model="gemini-2.0-flash").to_dict()

            # Disable AutoContext LLM calls by not generating titles/summaries
            auto_context_config = {
                "use_generated_title": False,
                "get_document_summary": False,
                "get_section_summaries": False,
            }

            # Perform add_document using existing images and serialized VLM
            kb.add_document(
                doc_id=doc_id,
                file_path="dummy.pdf",  # not used when images_already_exist=True
                document_title="Test Doc",
                auto_context_config=auto_context_config,
                file_parsing_config={
                    "use_vlm": True,
                    "vlm_config": {"images_already_exist": True},
                    "vlm": serialized_vlm,
                },
                semantic_sectioning_config={
                    "llm_provider": "openai",  # value not used when auto_context disabled
                    "model": "gpt-4o-mini",
                },
                chunking_config={},
            )

            # Assert elements.json exists
            elements_path = os.path.join(temp_dir, "page_images", kb_id, doc_id, "elements.json")
            self.assertTrue(os.path.exists(elements_path))

            # Assert page content jsons were created (convert_elements_to_page_content)
            page_content_files = glob(os.path.join(temp_dir, "page_images", kb_id, doc_id, "page_content_*.json"))
            self.assertTrue(len(page_content_files) >= 1)
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
