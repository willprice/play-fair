from abc import ABC

import numpy as np
import torch
from scipy.special import softmax

from models.components.mlp import MLPConsensus
from models.multiscale import MultiscaleModel
from attribution.characteristic_function_shapley_value_attributor import (
    CharacteristicFunctionExampleShapleyAttributor,
)
from attribution.naive_shapley_value_attributor import NaiveShapleyAttributor
from attribution.online_shapley_value_attributor import OnlineShapleyAttributor
from subset_samplers import ExhaustiveSubsetSampler
from combinatorics import powerset
from numpy.testing import assert_array_equal
from torch.distributions import Uniform


class BaseShapleyAttributorTest(ABC):
    n_classes = 21
    n_mlps = 4
    input_dim = 100

    @classmethod
    def setup_class(cls):
        np.random.seed(42)

        torch.random.manual_seed(42)
        cls.mlps = [
            MLPConsensus(cls.input_dim * i, 150, cls.n_classes, batch_norm=True)
            for i in range(1, cls.n_mlps + 1)
        ]
        for mlp in cls.mlps:
            mlp.eval()
        cls.priors = softmax(np.random.randn(cls.n_classes).astype(np.float32))
        cls.priors /= cls.priors.sum()

    def make_shapley_attributor(self):
        raise NotImplementedError()

    def test_efficiency_property_of_shapley_values(self):
        attributor = self.make_shapley_attributor()

        n_video_frames = len(self.mlps) + 2
        example = torch.randn(n_video_frames, self.input_dim)

        # shapley_values: (n_frames, n_classes)
        with torch.no_grad():
            shapley_values, scores = attributor.explain(example)
        shapley_values = shapley_values.numpy()
        scores = scores.numpy()

        class_scores = softmax(scores, axis=-1)
        np.testing.assert_allclose(
            shapley_values.sum(axis=0) + self.priors, class_scores[0], rtol=1e-5
        )

    def test_scores_against_multiscale_model(self):
        multiscale_model = MultiscaleModel(self.mlps, sampler=ExhaustiveSubsetSampler())
        multiscale_model.eval()
        shapley_attributor = self.make_shapley_attributor()
        n_video_frames = 5
        example = torch.randn(n_video_frames, self.input_dim)
        with torch.no_grad():
            model_scores = multiscale_model(example)
            shaley_values, attributor_scores = shapley_attributor.explain(example)

        np.testing.assert_allclose(
            attributor_scores.mean(axis=0).numpy(), model_scores.numpy(), rtol=1e-5
        )


class TestNaiveShapleyAttributor(BaseShapleyAttributorTest):
    def make_shapley_attributor(self):
        return NaiveShapleyAttributor(self.mlps, self.priors, self.n_classes)


class TestOnlineShapleyAttributor(BaseShapleyAttributorTest):
    def make_shapley_attributor(self):
        return OnlineShapleyAttributor(
            self.mlps,
            self.priors,
            n_classes=self.n_classes,
            subset_sampler=ExhaustiveSubsetSampler(),
        )

    def test_exhaustive_sampling_against_naive_implementation(self):
        online_attributor = OnlineShapleyAttributor(
            self.mlps, self.priors, self.n_classes,
                subset_sampler=ExhaustiveSubsetSampler()
        )

        n_video_frames = len(self.mlps) + 2
        example = torch.randn(n_video_frames, self.input_dim)

        online_shapley_values = online_attributor.explain(example)[0].numpy()

        naive_attributor = NaiveShapleyAttributor(self.mlps, self.priors, self.n_classes)
        naive_shapley_values = naive_attributor.explain(example)[0].numpy()

        np.testing.assert_allclose(
            online_shapley_values,
            naive_shapley_values,
            atol=1e-7,
            rtol=1e-7,
            verbose=True,
        )


class TestCharacteristicFunctionExampleShapleyAttributor:
    def test_efficiency_property(self):
        seq_features = torch.arange(3)  # [0, 1, 2]
        distribution = Uniform(0, 1)
        characteristic_fn_scores = {
            frozenset(subset): [distribution.sample().item()]
            for subset in powerset(seq_features.numpy())
        }
        characteristic_fn_scores[frozenset({})] = [0.0]

        def characteristic_fn(batch_seq_features):
            results = []
            for seq_features in batch_seq_features.numpy():
                results_key = frozenset(list(seq_features))
                results.append(characteristic_fn_scores[results_key])
            return torch.tensor(results, dtype=torch.float32)

        attributor = CharacteristicFunctionExampleShapleyAttributor(
            seq_features,
            characteristic_fn=characteristic_fn,
            iterations=1,
            subset_sampler=ExhaustiveSubsetSampler(),
            n_classes=1,
        )
        shapley_values, scores = attributor.run()
        assert_array_equal(
            shapley_values.sum(dim=0).numpy(),
            characteristic_fn_scores[frozenset(list(seq_features.numpy()))],
        )


def assert_ordered(xs: tuple):
    if len(xs) <= 1:
        return
    prev = xs[0]
    for i, next in enumerate(xs[1:], start=1):
        assert next >= prev, f"xs[{i}] < xs[{i - 1}]"
