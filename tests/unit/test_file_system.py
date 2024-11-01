import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.dsparse.file_parsing.file_system import LocalFileSystem, S3FileSystem
from dsrag.knowledge_base import KnowledgeBase


class TestFileSystem(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.base_path = "~/dsrag_test_file_system"
        self.local_file_system = LocalFileSystem(base_path=self.base_path)
        self.s3_file_system = S3FileSystem(
            base_path=self.base_path,
            bucket_name=os.environ["AWS_S3_BUCKET_NAME"],
            region_name=os.environ["AWS_S3_REGION"],
            access_key=os.environ["AWS_S3_ACCESS_KEY"],
            secret_key=os.environ["AWS_S3_SECRET_KEY"]
        )
        self.kb_id = "test_file_system_kb"
        self.kb = KnowledgeBase(kb_id=self.kb_id, file_system=self.local_file_system)

    def test__001_load_kb(self):
        kb = KnowledgeBase(kb_id=self.kb_id)
        # Assert the class
        assert isinstance(kb.file_system, LocalFileSystem)

    def test__002_overwrite_file_system(self):
        kb = KnowledgeBase(kb_id=self.kb_id, file_system=self.s3_file_system)
        # Assert the class. We don't want to allow overwriting the file system
        assert isinstance(kb.file_system, LocalFileSystem)

    @classmethod
    def tearDownClass(self):
        try:
            os.system(f"rm -rf {self.base_path}")
        except:
            pass


if __name__ == "__main__":
    unittest.main()