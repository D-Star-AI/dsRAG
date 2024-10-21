import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dsrag.dsparse.chunking import find_lines_in_range, chunk_sub_section


class TestDsParse(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dsparse_output'))
    
    def test__find_lines_in_range(self):
        pass

    def test__chunk_sub_section(self):
        pass

if __name__ == "__main__":
    unittest.main()