import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal
from scipy.special import comb

from tensor_ops import sum_all_but, random_subsequence_indices, masked_mean, \
    masked_sum


def torch_test_tensor(*shape):
    return torch.arange(int(np.product(shape))).reshape(*shape)


def test_sum_all_but_last_dim():
    xs = torch_test_tensor(3, 2, 10)
    index = 3
    summed = sum_all_but(xs, index, dim=-1)
    assert summed.ndimension() == xs.ndimension()
    assert (xs[..., index] == summed[..., 0]).all()
    expected_sum = (xs.sum(dim=-1) - xs[..., index])
    assert (expected_sum - summed[..., 1]).sum().item() < 1e-7


def test_sum_all_but_middle_dim():
    xs = torch_test_tensor(3, 10, 2)
    index = 3
    summed = sum_all_but(xs, index, dim=1)
    assert summed.ndimension() == xs.ndimension()
    assert (xs[..., index, :] == summed[..., 0, :]).all()
    expected_sum = (xs.sum(dim=1) - xs[..., index, :])
    assert (expected_sum - summed[..., 1, :]).sum().item() < 1e-8


@given(st.data())
def test_random_subsequence_indices(data):
    sequence_length = data.draw(st.integers(min_value=1, max_value=10))
    subsequence_length = data.draw(st.integers(min_value=1, max_value=sequence_length))
    sample_size = data.draw(st.integers(min_value=1, max_value=1000))
    sample_idx = random_subsequence_indices(sequence_length, subsequence_length,
                                            sample_size=sample_size)
    assert sample_idx.shape == (sample_size, subsequence_length)
    assert isinstance(sample_idx, torch.LongTensor)
    assert (sample_idx <= (sequence_length - 1)).all()
    assert (sample_idx >= 0).all()
    # check ordering
    if sequence_length > 1:
        assert ((sample_idx[:, 1:] - sample_idx[:, :-1]) > 0).all()


@given(st.data())
def test_random_subsequence_indices_without_duplicates(data):
    sequence_length = data.draw(st.integers(min_value=1, max_value=10))
    subsequence_length = data.draw(st.integers(min_value=1, max_value=sequence_length))
    sample_size = data.draw(st.integers(min_value=1, max_value=1000))
    sample_idx = random_subsequence_indices(sequence_length, subsequence_length,
                                            sample_size=sample_size,
                                            exclude_repeats=True)
    expected_size = min(comb(sequence_length, subsequence_length, exact=True), sample_size)
    assert sample_idx.shape == (expected_size, subsequence_length)
    assert isinstance(sample_idx, torch.LongTensor)
    assert (sample_idx <= (sequence_length - 1)).all()
    assert (sample_idx >= 0).all()
    # check ordering
    if sequence_length > 1:
        assert ((sample_idx[:, 1:] - sample_idx[:, :-1]) > 0).all()
    samples = {
        frozenset(s) for s in
        sample_idx
    }
    assert len(samples) == len(sample_idx)


def test_masked_sum():
    mask = torch.tensor([
        [True, False, False],
        [False, False, True],
        [True, True, False],
        [False, False, False],
    ], dtype=torch.bool)
    # (3, 2)
    data = torch.tensor([
        [1, 2],
        [2, 3],
        [5, 1],
    ], dtype=torch.float)

    sum = masked_sum(mask, data, nan_replacement=True)
    assert_array_equal(sum.numpy(), np.array([
        [1, 2],
        [5, 1],
        [3, 5],
        [np.nan, np.nan]
    ], dtype=np.float))

    sum = masked_sum(mask, data, nan_replacement=False)
    assert_array_equal(sum.numpy(), np.array([
        [1, 2],
        [5, 1],
        [3, 5],
        [0, 0]
    ], dtype=np.float))


def test_masked_mean():
    # (4, 3)
    mask = torch.tensor([
        [True, False, False],
        [False, False, True],
        [True, True, False],
        [False, False, False],
    ], dtype=torch.bool)
    # (3, 2)
    data = torch.tensor([
        [1, 2],
        [2, 3],
        [5, 1],
    ], dtype=torch.float)
    mean = masked_mean(mask, data, nan_replacement=True)
    assert_array_equal(mean.numpy(), np.array([
        [1, 2],
        [5, 1],
        [1.5, 2.5],
        [np.nan, np.nan]
    ]))

    mean = masked_mean(mask, data, nan_replacement=False)
    assert_array_equal(mean.numpy(), np.array([
        [1, 2],
        [5, 1],
        [1.5, 2.5],
        [0, 0]
    ]))


