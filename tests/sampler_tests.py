import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal
from torchvideo.samplers import frame_idx_to_list

from frame_sampling import sample_uniform_idx, sample_uniformly, RandomSampler


@given(st.data())
def test_sampling_uniform_idx_has_mean_half_array_length(data):
    array_length = data.draw(st.integers(min_value=1, max_value=1e6),
                             label="array_length")
    sample_count = data.draw(st.integers(min_value=1, max_value=array_length),
                             label="sample_count")
    bound = array_length / 2
    # +- 0.5 bounds since we can't always get perfectly on the mean
    # we use 0.51 as rounding errors creep in for massive arrays and can make the mean
    # slightly above bound + 0.5 or below bound - 0.5
    assert bound - 0.51 <= np.mean(sample_uniform_idx(array_length, sample_count)) <= \
           bound + 0.51


@pytest.mark.parametrize("array,sample_count,expected_array",[
    ([0, 1, 2], 1, [1]),
    ([0, 1, 2], 2, [0, 2]),
    ([0, 1, 2, 3], 2, [1, 3]),
    ([0, 1, 2, 3, 4], 1, [2]),
    ([0, 1, 2, 3, 4], 2, [1, 3]),
    ([0, 1, 2, 3, 4], 3, [0, 2, 4]),
    ([0, 1, 2, 3, 4], 3, [0, 2, 4]),
    ([0, 1, 2, 3, 4, 5], 3, [1, 3, 5]),
    ([0, 1, 2, 3, 4, 5, 6], 3, [1, 3, 5]),
    ([0, 1, 2, 3, 4, 5, 6, 7, 8], 3, [1, 4, 7]),
])
def test_uniform_sampler(array, sample_count, expected_array):
    assert_array_equal(sample_uniformly(np.array(array), sample_count),
                       np.array(expected_array))


@given(st.data())
def test_random_sampler(data):
    video_length = data.draw(st.integers(min_value=1, max_value=1e3),
                             label="Video length")
    snippet_length = data.draw(st.integers(min_value=1, max_value=video_length),
                               label="Snippet length")
    frame_sample_count = data.draw(st.integers(min_value=1, max_value=100),
                             label="Frame sample count")
    test = data.draw(st.booleans(), label="Test mode?")
    sampler = RandomSampler(frame_sample_count, snippet_length, test=test)

    samples = frame_idx_to_list(sampler.sample(video_length))

    assert len(samples) == snippet_length * frame_sample_count
