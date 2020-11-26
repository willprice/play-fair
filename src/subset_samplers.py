import random

from abc import ABC
from collections import defaultdict
from typing import FrozenSet
from typing import Iterable
from typing import Optional
from typing import Set
from typing import TypeVar

import numpy as np
import torch

from tensor_ops import random_subsequence_indices
from tensor_ops import subsequence_indices
from scipy.special import comb


class SubsetSampler(ABC):
    device: torch.device

    def sample(self, n_video_frames, n_frames):
        raise NotImplementedError()


class ExhaustiveSubsetSampler(SubsetSampler):
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device

    def sample(self, n_video_frames, n_frames) -> torch.Tensor:
        assert n_video_frames >= 1
        assert 0 <= n_frames <= n_video_frames
        return subsequence_indices(n_video_frames, n_frames, device=self.device)


class RandomSubsetSampler(SubsetSampler):
    def __init__(
        self,
        max_samples: int,
        device: Optional[torch.device] = None,
        exclude_repeats: bool = False,
    ):
        self.device = device
        self.max_samples = max_samples
        self.exclude_repeats = exclude_repeats

    def sample(self, n_video_frames, n_frames):
        assert n_video_frames >= 1
        assert 0 <= n_frames <= n_video_frames
        return random_subsequence_indices(
            n_video_frames,
            n_frames,
            sample_size=self.max_samples,
            exclude_repeats=self.exclude_repeats,
            device=self.device,
        )


class RandomProportionSubsetSampler(SubsetSampler):
    def __init__(
        self,
        p: float,
        min_samples: int,
        exclude_repeats: bool = True,
        device: Optional[torch.device] = None,
    ):
        assert 0 < p <= 1
        self.p = p
        self.min_samples = min_samples
        self.exclude_repeats = exclude_repeats
        self.device = device

    def sample(self, n_video_frames, n_frames):
        total_combinations = comb(n_video_frames, n_frames, exact=True)
        sample_size = int(max(total_combinations * self.p, self.min_samples))
        return random_subsequence_indices(
            n_video_frames,
            n_frames,
            sample_size=sample_size,
            exclude_repeats=self.exclude_repeats,
            device=self.device,
        )


class RandomSubsetSamplerWithoutRepeats(SubsetSampler):
    def __init__(
        self, max_samples: int, device: Optional[torch.device] = None,
    ):
        self.seen = defaultdict(lambda: set())
        self.max_samples = max_samples
        self.device = device

    def sample(self, n_video_frames, n_frames):
        assert n_video_frames >= 1
        assert 0 <= n_frames <= n_video_frames
        population = list(range(n_video_frames))
        seen = self.seen[(n_video_frames, n_frames)]
        n_possible_samples = comb(n_video_frames, n_frames, exact=True)
        n_remaining_items = n_possible_samples - len(seen)
        sample_size = min(n_remaining_items, self.max_samples)

        if n_frames == 0:
            return torch.zeros((1, 0), dtype=torch.long, device=self.device)
        if sample_size == 0:
            return (
                torch.tensor([], dtype=torch.long)
                .reshape((0, n_frames))
                .to(self.device)
            )
        samples = set()
        while len(samples) != sample_size:
            sample = frozenset(random.sample(population, k=n_frames))
            if sample not in seen:
                samples.add(sample)
                seen.add(sample)

        return torch.tensor(
            [sorted(s) for s in samples], dtype=torch.int, device=self.device
        )

    def reset(self):
        for key in self.seen.keys():
            self.seen[key] = set()


class AbstractConstructiveRandomSampler(SubsetSampler, ABC):
    """Samples new subsets based on iteratively adding new items to previously
    sampled subsets"""

    def __init__(
        self, max_samples: int, device: Optional[torch.device] = None,
    ):
        self.device = device
        self.max_samples = max_samples
        self.reset()

    def reset(self):
        self.n_video_frames = None
        self.population = None
        self.previous_n_frames = 0
        self.previous_subsets = {frozenset()}

    def sample(self, n_video_frames, n_frames):
        assert n_video_frames >= 1, f"n_video_frames ({n_video_frames}) >= 1"
        assert 0 <= n_frames <= n_video_frames, (
            f"0 <= n_frames ({n_frames}) <= " f"n_video_frames ({n_video_frames})"
        )
        if n_frames == 0:
            return torch.zeros((1, 0), dtype=torch.long, device=self.device)
        assert (
            n_frames == self.previous_n_frames + 1
        ), f"n_frames ({n_frames}) == self.previous_n_frames + 1 ({self.previous_n_frames + 1})"
        if self.n_video_frames is None:
            self.n_video_frames = n_video_frames
        if self.population is None:
            self.population = set(range(n_video_frames))
        assert (
            self.n_video_frames == n_video_frames
        ), f"self.n_video_frames ({self.n_video_frames}) == n_video_frames ({n_video_frames})"

        subsets = self._sample(n_video_frames, n_frames)

        self.previous_subsets = subsets
        self.previous_n_frames = n_frames

        return torch.tensor(
            sorted([sorted(list(s)) for s in subsets]), device=self.device
        ).to(torch.long)

    def _sample(self, n_video_frames: int, n_frames: int) -> Set[FrozenSet[int]]:
        raise NotImplementedError()


class ConstructiveRandomSampler(AbstractConstructiveRandomSampler):
    """Samples new subsets based on iteratively adding new items to previously
    sampled subsets"""

    def _sample(self, n_video_frames: int, n_frames: int) -> Set[FrozenSet[int]]:
        max_possible_samples = comb(n_video_frames, n_frames, exact=True)
        candidates = ConstructiveRandomSampler.compute_candidate_pool(
            self.population, self.previous_subsets
        )
        n = min(len(candidates), max_possible_samples, self.max_samples)
        return set(random.sample(candidates, n))

    @staticmethod
    def compute_candidate_pool(population: Set[int], subsets: Set[FrozenSet[int]]):
        """Compute the set of candidates that you can sample by growing the subsets
        in ``subsets`` by 1 element selected from population"""
        candidates = {
            to_extend.union({new_element})
            for to_extend in subsets
            for new_element in population - to_extend
        }
        return candidates


class ProportionalConstructiveRandomSampler(AbstractConstructiveRandomSampler):
    """Samples new subsets based on iteratively adding new items to previously
    sampled subsets"""

    def __init__(
        self, p: float, min_samples: int = 1, device: Optional[torch.device] = None,
    ):
        assert 0 < p <= 1
        self.device = device
        self.min_samples = min_samples
        self.p = p
        self.reset()

    def _sample(self, n_video_frames: int, n_frames: int) -> Set[FrozenSet[int]]:
        max_possible_samples = comb(n_video_frames, n_frames, exact=True)
        n_to_sample = max(
            int(np.round(max_possible_samples * self.p)), self.min_samples
        )
        candidates = ConstructiveRandomSampler.compute_candidate_pool(
            self.population, self.previous_subsets
        )
        n = min(len(candidates), max_possible_samples, n_to_sample)
        return set(random.sample(candidates, n))

    @staticmethod
    def compute_candidate_pool(population: Set[int], subsets: Set[FrozenSet[int]]):
        """Compute the set of candidates that you can sample by growing the subsets
        in ``subsets`` by 1 element selected from population"""
        candidates = {
            to_extend.union({new_element})
            for to_extend in subsets
            for new_element in population - to_extend
        }
        return candidates


T = TypeVar("T")


def union(sets: Iterable[Set[T]]) -> Set[T]:
    unioned = set()
    for s in sets:
        unioned.update(s)
    return unioned
