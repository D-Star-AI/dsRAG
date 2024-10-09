import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.dsparse.chunking import chunk_document, chunk_sub_section

class TestChunking(unittest.TestCase):
    def test__chunk_document(self):
        pass

    def test__chunk_sub_section(self):
        pass

if __name__ == "__main__":
    unittest.main()