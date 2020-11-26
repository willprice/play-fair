import random

from typing import List
from typing import Optional
from typing import Union

import numpy as np

from torchvideo.samplers import FrameSampler


class RandomSampler(FrameSampler):
    def __init__(self, frame_count: int, snippet_length: int, test: bool = False):
        self.frame_count = frame_count
        self.snippet_length = snippet_length
        self.test = test
        self.choice_idx = np.arange(frame_count)

    def sample(self, video_length: int) -> Union[slice, List[int], List[slice]]:
        if video_length < self.frame_count * self.snippet_length:
            return generate_oversampled_idx(
                video_length, self.frame_count, self.snippet_length
            )
        if self.test:
            return self.sample_test(video_length)
        return self.sample_train(video_length)

    def sample_test(self, video_length):
        return [
            slice(start_pos, start_pos + self.snippet_length)
            for start_pos in sample_uniform_idx(video_length, self.frame_count)
        ]

    def sample_train(self, video_length):
        possible_sample_positions = np.arange(
            video_length + 1 - self.snippet_length * self.frame_count
        )
        unordered_position_idx = np.random.choice(
            np.arange(len(possible_sample_positions)), size=self.frame_count
        )
        starting_position_idx = np.sort(np.array(unordered_position_idx))
        starting_positions = possible_sample_positions[starting_position_idx]
        snippet_offsets = np.arange(
            0, self.snippet_length * self.frame_count, self.snippet_length
        )
        return [
            slice(position, position + self.snippet_length)
            for position in starting_positions + snippet_offsets
        ]


def sample_uniformly(array: np.array, sample_count: int) -> np.array:
    sample_idx = sample_uniform_idx(len(array), sample_count)
    return array[sample_idx]


def sample_uniform_idx(array_length: int, sample_count: int) -> np.array:
    segment_length = array_length / sample_count
    segment_idx = np.arange(sample_count)
    segment_starts = segment_idx * segment_length
    sampling_idx = segment_starts + segment_length / 2
    sampling_idx = np.floor(sampling_idx).astype(np.uint32)
    return sampling_idx


def generate_oversampled_idx(video_length: int, frame_count: int, snippet_length: int):
    return [0] * (frame_count * snippet_length)


def generate_segments(seq_len: int, n_segments: int) -> np.ndarray:
    """
    Returns:
        segment [start, stop) indices of shape :math:`(N, 2)`
    """
    if seq_len < n_segments:
        raise ValueError(
            f"seq_len ({seq_len}) < n_segments ({n_segments}); can't split into segments."
        )
    segment_length = seq_len / n_segments
    start_idxs = np.arange(n_segments) * segment_length
    stop_idxs = start_idxs + segment_length
    return np.stack([np.round(start_idxs), np.round(stop_idxs)], axis=-1).astype(
        np.int32
    )


def random_sample_segments(segments: np.ndarray) -> np.ndarray:
    """
    Args:
        segments: Array of shape :math:`(N, 2)` containing segment bounds for each of :math:`N` segments.
            Lower bound is inclusive, upper is exclusive.
    Returns:
        random sample of frames selected from each segment.
    """
    idxs = [np.random.randint(low=low, high=high) for low, high in segments]
    return np.stack(idxs)


def center_sample_segments(segments: np.ndarray) -> np.ndarray:
    """
    Args:
        segments: Array of shape :math:`(N, 2)` containing segment bounds for each of :math:`N` segments.
            Lower bound is inclusive, upper is exclusive.
    Returns:
        Center indices from each segment
    """
    seq_len = segments[-1, 1]
    n_segments = len(segments)
    segment_width = seq_len / n_segments
    center_idx = np.floor(
        np.arange(n_segments) * segment_width + segment_width / 2
    ).astype(np.int32)
    return center_idx
