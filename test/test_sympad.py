import pytest

import torch
import numpy as np
from sympad import _pad_symmetric_1d, _pad_symmetric


@pytest.mark.parametrize("size", [[5], [6], [9], [3]])
@pytest.mark.parametrize("pad_list", [(1, 4), (2, 2), (3, 3), (4, 1), (5, 0), (0, 5), (0, 0), (1, 1), (3, 1), (1, 3)])
def test_pad_symmetric_1d(size: list[int], pad_list: tuple[int, int]) -> None:
    """Test high-dimensional symetric padding."""
    array = np.random.randint(0, 9, size=size)
    my_pad = _pad_symmetric_1d(torch.from_numpy(array), pad_list)
    np_pad = np.pad(array, pad_list, mode="symmetric")
    assert np.allclose(my_pad.numpy(), np_pad)



@pytest.mark.parametrize("size", [[6, 5], [5, 6], [5, 5], [9, 9], [3, 3], [4, 4]])
@pytest.mark.parametrize("pad_list", [[(1, 4), (4, 1)], [(2, 2), (3, 3)]])
def test_pad_symmetric_2d(size: list[int], pad_list: list[tuple[int, int]]) -> None:
    """Test high-dimensional symetric padding."""
    array = np.random.randint(0, 9, size=size)
    my_pad = _pad_symmetric(torch.from_numpy(array), pad_list)
    np_pad = np.pad(array, pad_list, mode="symmetric")
    assert np.allclose(my_pad.numpy(), np_pad)

@pytest.mark.parametrize("size", [[3, 6, 5], [1, 6, 7]])
@pytest.mark.parametrize("pad_list", [[(0, 0), (1, 4), (4, 1)], [(1,1), (2, 2), (3, 3)]])
def test_pad_symmetric_3d(size: list[int], pad_list: list[tuple[int, int]]) -> None:
    """Test high-dimensional symetric padding."""
    array = np.random.randint(0, 9, size=size)
    my_pad = _pad_symmetric(torch.from_numpy(array), pad_list)
    np_pad = np.pad(array, pad_list, mode="symmetric")
    assert np.allclose(my_pad.numpy(), np_pad)


def test_pad_symmetric_small() -> None:
    """Test high-dimensional symetric padding."""
    array = np.random.randint(0, 9, size=(2,2))
    my_pad = _pad_symmetric(torch.from_numpy(array), ((1,1), (1,1)))
    np_pad = np.pad(array, ((1,1), (1,1)), mode="symmetric")
    assert np.allclose(my_pad.numpy(), np_pad)