import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dsrag.semantic_sectioning import partition_sections, is_valid_partition, Section


class TestPartitionSections(unittest.TestCase):
    # Test case 1: Sections already form a complete partition
    def test__complete_partition(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="Body", start_index=5, end_index=9),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        a = 0
        b = 14
        expected = sections
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 2: Gap at the beginning
    def test__gap_at_beginning(self):
        sections = [
            Section(title="Body", start_index=5, end_index=9),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        a = 0
        b = 14
        expected = [
            Section(title="", start_index=0, end_index=4),
            Section(title="Body", start_index=5, end_index=9),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 3: Gap at the end
    def test__gap_at_end(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="Body", start_index=5, end_index=9)
        ]
        a = 0
        b = 14
        expected = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="Body", start_index=5, end_index=9),
            Section(title="", start_index=10, end_index=14)
        ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 4: Gap in the middle
    def test__gap_in_middle(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        a = 0
        b = 14
        expected = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="", start_index=5, end_index=9),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"
    
    # Test case 5: Multiple gaps in the middle
    def test__multiple_gaps_in_middle(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="Middle", start_index=10, end_index=14),
            Section(title="Conclusion", start_index=20, end_index=24)
        ]
        a = 0
        b = 24
        expected = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="", start_index=5, end_index=9),
            Section(title="Middle", start_index=10, end_index=14),
            Section(title="", start_index=15, end_index=19),
            Section(title="Conclusion", start_index=20, end_index=24)
        ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 6: Overlapping sections
    def test__overlapping_sections(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=6),
            Section(title="Body", start_index=5, end_index=9),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        a = 0
        b = 14
        expected = [
            Section(title="", start_index=0, end_index=4),
            Section(title="Body", start_index=5, end_index=9),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"
    
    # Test case 7: Gap, then overlapping section
    def test__gap_then_overlapping_section(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=3),
            Section(title="Body", start_index=7, end_index=11),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        a = 0
        b = 14
        expected = [
            Section(title="Introduction", start_index=0, end_index=3),
            Section(title="", start_index=4, end_index=6),
            Section(title="", start_index=7, end_index=9),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 8: Overlapping then gap
    def test__overlapping_then_gap(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=6),
            Section(title="Body", start_index=5, end_index=9)
        ]
        a = 0
        b = 14
        expected = [
            Section(title="", start_index=0, end_index=4),
            Section(title="Body", start_index=5, end_index=9),
            Section(title="", start_index=10, end_index=14)
        ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 9: One section is a subset of another section
    def test__subset_sections(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=10),
            Section(title="Subset", start_index=5, end_index=9)
        ]
        a = 0
        b = 14
        expected = [
            Section(title="Introduction", start_index=0, end_index=10),
            Section(title="", start_index=11, end_index=14)
        ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 10: Sections with the same start and end indices
    def test__same_indices(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=10),
            Section(title="Introduction", start_index=0, end_index=10)
        ]
        a = 0
        b = 14
        expected = [
            Section(title="Introduction", start_index=0, end_index=10),
            Section(title="", start_index=11, end_index=14)
        ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 11: Sections completely outside the range [a, b]
    def test__sections_outside(self):
        sections = [
            Section(title="Outside", start_index=15, end_index=20)
        ]
        a = 0
        b = 14
        expected = [
                Section(title="", start_index=a, end_index=b)
            ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 12: Sections with invalid indices
    def test__invalid_indices(self):
        sections = [
            Section(title="Invalid", start_index=10, end_index=9)
        ]
        a = 0
        b = 14
        expected = [
                Section(title="", start_index=a, end_index=b)
            ]
        result = partition_sections(sections, a, b)
        assert result == expected, f"Expected {expected}, but got {result}"


class TestIsValidPartition(unittest.TestCase):

    # Test case 1: Valid partition
    def test__valid_partition(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="Body", start_index=5, end_index=9),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        a = 0
        b = 14
        result = is_valid_partition(sections, a, b)
        assert result == True
    
    # Test case 2: Invalid partition due to gap
    def test__invalid_partition(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="Body", start_index=5, end_index=8),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        a = 0
        b = 14
        result = is_valid_partition(sections, a, b)
        assert result == False
    
    # Test case 3: Invalid partition due to start index not matching
    def test__invalid_partition_2(self):
        sections = [
            Section(title="Introduction", start_index=1, end_index=4),
            Section(title="Body", start_index=5, end_index=9),
            Section(title="Conclusion", start_index=10, end_index=14)
        ]
        a = 0
        b = 14
        result = is_valid_partition(sections, a, b)
        assert result == False
    

    # Test case 4: Invalid partition due to end index not matching
    def test__invalid_partition_3(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="Body", start_index=5, end_index=9),
            Section(title="Conclusion", start_index=10, end_index=13)
        ]
        a = 0
        b = 14
        result = is_valid_partition(sections, a, b)
        assert result == False
    

    # Test case 5: Invalid partition due to overlapping sections
    def test__invalid_partition_4(self):
        sections = [
            Section(title="Introduction", start_index=0, end_index=4),
            Section(title="Body", start_index=5, end_index=9),
            Section(title="Conclusion", start_index=8, end_index=14)
        ]
        a = 0
        b = 14
        result = is_valid_partition(sections, a, b)
        assert result == False


# Run all tests
if __name__ == '__main__':
    unittest.main()