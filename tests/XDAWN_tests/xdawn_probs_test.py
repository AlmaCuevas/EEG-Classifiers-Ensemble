import pytest

from xdawn_probs import xdawn_test

expected = 2
result = 1

def test_xdawn_test() -> None:
    xdawn_test()
    assert expected == result
