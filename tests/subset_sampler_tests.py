import torch

from hypothesis import given
from hypothesis import strategies as st
from subset_samplers import ConstructiveRandomSampler
from subset_samplers import ExhaustiveSubsetSampler
from subset_samplers import ProportionalConstructiveRandomSampler
from subset_samplers import RandomProportionSubsetSampler
from subset_samplers import RandomSubsetSampler
from subset_samplers import RandomSubsetSamplerWithoutRepeats
from tensor_ops import compute_subset_relations


def assert_no_repeats(collection_of_collections):
    assert len(set(map(frozenset, collection_of_collections))) == len(
        collection_of_collections
    )


class BaseSubsetSamplerTests:
    @given(st.data())
    def test_sampler_generates_ordered_idx(self, data):
        sampler = self.make_sampler(data)
        n_video_frames, n_frames = self.draw_sampling_parameters(data)
        sample_idxs = sampler.sample(n_video_frames, n_frames)
        for sample_idx in sample_idxs:
            assert sorted(sample_idx) == list(sample_idx)

    @given(st.data())
    def test_sampling_0_elements(self, data):
        sampler = self.make_sampler(data)
        sample = sampler.sample(12, 0)
        assert sample.shape == (1, 0)
        assert sample.dtype == torch.long

    def draw_sampling_parameters(self, data):
        n_video_frames = data.draw(st.integers(min_value=1, max_value=12))
        n_frames = data.draw(st.integers(min_value=1, max_value=n_video_frames))
        return n_video_frames, n_frames

    def make_sampler(self, data):
        raise NotImplementedError()


class TestExhaustiveSubsetSampler(BaseSubsetSamplerTests):
    def make_sampler(self, data):
        return ExhaustiveSubsetSampler()


class TestRandomSubsetSampler(BaseSubsetSamplerTests):
    def make_sampler(self, data):
        return RandomSubsetSampler(
            max_samples=20, exclude_repeats=data.draw(st.booleans())
        )

    @given(st.data())
    def test_no_repeats_when_exclude_repeats_is_set(self, data):
        sampler = RandomSubsetSampler(max_samples=20, exclude_repeats=True)
        n_video_frames, n_frames = self.draw_sampling_parameters(data)
        sample_idxs = sampler.sample(n_video_frames, n_frames)
        assert_no_repeats(sample_idxs)


class TestRandomProportionSubsetSampler(BaseSubsetSamplerTests):
    def make_sampler(self, data):
        return RandomProportionSubsetSampler(
            p=data.draw(st.floats(min_value=1e-5, max_value=1)),
            min_samples=1,
            exclude_repeats=data.draw(st.booleans()),
        )


class TestRandomSubsetSamplerWithoutRepeats(BaseSubsetSamplerTests):
    def make_sampler(self, data):
        return RandomSubsetSamplerWithoutRepeats(
            max_samples=data.draw(st.integers(min_value=1, max_value=20))
        )


class BaseConstructiveSamplerTests:
    def _make_sampler(self):
        return ConstructiveRandomSampler(max_samples=20)

    def test_sampling_0_elements(self):
        sampler = self._make_sampler()
        sample = sampler.sample(12, 0)
        print(sample)
        assert sample.shape == (1, 0)

    def test_sampler_generates_ordered_idx(self):
        sampler = self._make_sampler()
        samples = []
        n_frames = 12
        for scale in range(1, n_frames + 1):
            samples.extend(sampler.sample(n_video_frames=12, n_frames=scale))

        for sample in samples:
            assert sorted(sample) == list(sample)

    def test_sampler_builds_on_previous_subsets(self):
        sampler = self._make_sampler()
        samples = self.sample_all_scales(sampler)

        for previous_samples, current_samples in zip(samples, samples[1:]):
            subset_relations = compute_subset_relations(
                current_samples, previous_samples
            )
            assert subset_relations.any(-1).all()

    def test_sampler_does_not_have_repeats(self):
        sampler = self._make_sampler()
        samples = self.sample_all_scales(sampler)
        for scale_samples in samples:
            assert_no_repeats(scale_samples)

    def sample_all_scales(self, sampler):
        samples = []
        max_set_size = 12
        for sample_size in range(1, max_set_size + 1):
            samples.append(sampler.sample(max_set_size, sample_size))
        return samples


class TestConstructiveRandomSampler(BaseConstructiveSamplerTests):
    def _make_sampler(self):
        return ConstructiveRandomSampler(max_samples=20)


class TestProportionalConstructiveRandomSampler(BaseConstructiveSamplerTests):
    def _make_sampler(self):
        return ProportionalConstructiveRandomSampler(p=0.1)
