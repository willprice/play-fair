import numpy as np
import torch

from models.components.mlp import MLPConsensus
from models.multiscale import MultiscaleModel
from models.multiscale import RecursiveMultiscaleModel
from subset_samplers import ExhaustiveSubsetSampler


class TestRecursiveMultiscaleModel:
    n_classes = 20
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
        cls.priors = np.random.randn(cls.n_classes)
        cls.priors /= cls.priors.sum()

    def test_against_results_are_the_same_as_the_multiscale_model(self):
        sampler = ExhaustiveSubsetSampler()
        multiscale_model = MultiscaleModel(
            self.mlps, softmax=False, sampler=sampler, save_intermediate=True
        )
        recursive_multiscale_model = RecursiveMultiscaleModel(
            self.mlps,
            sampler=sampler,
            save_intermediate=True,
            count_n_evaluations=False,
        )
        for n_video_frames in range(1, len(self.mlps) + 8):
            example = torch.randn(n_video_frames, self.input_dim)

            with torch.no_grad():
                multiscale_model_results = multiscale_model(example.clone()).numpy()
            with torch.no_grad():
                recursive_multiscale_model_results = recursive_multiscale_model(
                    example.clone()
                ).numpy()

            np.testing.assert_allclose(
                recursive_multiscale_model_results,
                multiscale_model_results,
                err_msg=f"Failure comparing scores for a {n_video_frames} frame input",
                rtol=1e-4,
            )
