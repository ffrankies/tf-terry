import pytest
from tensorflow_rnn.batchmaker import *

SORTED_DATA = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3],
    [0, 1, 2],
    [0, 1],
    [0]
]

SCRAMBLED_DATA = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2],
    [0, 1],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3],
    [0],
    [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5]
]

class TestSortByLength():
    def test_should_not_change_data_when_it_is_empty(self):
        assert sort_by_length([]) == ()
    
    def test_should_not_change_data_when_it_is_already_sorted(self):
        assert sort_by_length([SORTED_DATA]) == tuple([SORTED_DATA])

    def test_should_sort_scrambled_data(self):
        assert sort_by_length([SCRAMBLED_DATA]) == tuple([SORTED_DATA])

    def test_should_sort_all_data(self):
        assert sort_by_length([SORTED_DATA, SCRAMBLED_DATA, SCRAMBLED_DATA, SORTED_DATA]) == (
            SORTED_DATA, SORTED_DATA, SORTED_DATA, SORTED_DATA)