from itertools import combinations
from typing import cast
from typing import Tuple

import numpy as np
import torch.nn

from tensor_ops import masked_mean


class NaiveShapleyAttributor:
    def __init__(self, single_scale_models, priors, n_classes, device=None):
        self.single_scale_models = single_scale_models
        self.priors = torch.from_numpy(priors).reshape(1, -1)
        self.device = device
        self.n_classes = n_classes

    def explain(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_video_frames = video.shape[0]
        n_mlps = len(self.single_scale_models)

        previous_scale_scores = self.priors.clone()
        previous_scale_frame_idxs = cast(
            torch.LongTensor, torch.zeros((1, 0), dtype=torch.long, device=self.device)
        )
        # shapley_contributions[scale_index, frame_index] -> shapley contributions
        # for frame `frame_index` from scale `scale_index`
        shapley_contributions = torch.zeros(
            (n_video_frames, n_video_frames, self.n_classes),
            dtype=torch.float32,
            device=self.device,
        )

        for scale_idx in range(n_video_frames):
            n_frames = scale_idx + 1
            current_scale_frame_idxs = self.sample_subsequence_idxs(
                n_video_frames, n_frames
            )
            subset_relations = self.compute_subset_relationships(
                current_scale_frame_idxs, previous_scale_frame_idxs
            )

            subsequences = video[current_scale_frame_idxs]
            if scale_idx < n_mlps:
                mlp = self.single_scale_models[scale_idx]
                with torch.no_grad():
                    current_scale_scores = mlp(subsequences)

                if n_frames >= 2:
                    current_scale_scores = self.combine_scores(
                        current_scale_scores,
                        previous_scale_scores,
                        subset_relations,
                        n_frames,
                    )
            else:
                current_scale_scores = masked_mean(
                    subset_relations,
                    previous_scale_scores,
                    bs=subset_relations.shape[0],
                )
            shapley_contributions[scale_idx] = self.compute_shapley_contributions(
                current_scale_scores,
                current_scale_frame_idxs,
                previous_scale_scores,
                previous_scale_frame_idxs,
                n_video_frames,
                n_frames,
            )
            previous_scale_scores = current_scale_scores
            previous_scale_frame_idxs = current_scale_frame_idxs

        shapley_values = shapley_contributions.mean(dim=0)
        return shapley_values, current_scale_scores

    def sample_subsequence_idxs(
        self, n_total_elements: int, n_sample_elements: int
    ) -> torch.LongTensor:
        """

        Args:
            n_total_elements: :math:`n` Number of total elements in the sequence.
            n_sample_elements: :math:`r` Number of elements to sample.

        Returns:
            Tensor of shape :math:`(b, n)`  where :math:`b` is :math:`n \\choose r`
        """
        assert n_sample_elements <= n_total_elements
        return cast(
            torch.LongTensor,
            torch.tensor(
                list(combinations(np.arange(n_total_elements), n_sample_elements)),
                dtype=torch.long,
                device=self.device,
            ),
        )

    def combine_scores(
        self,
        current_scale_scores: torch.FloatTensor,
        previous_scale_scores: torch.FloatTensor,
        subset_relations: torch.BoolTensor,
        n_frames: int,
        inplace: bool = True,
    ):
        if not inplace:
            current_scale_scores = current_scale_scores.clone()

        avg_previous_scale_scores = masked_mean(
            subset_relations, previous_scale_scores, bs=len(subset_relations)
        )
        current_scale_scores.add_(avg_previous_scale_scores, alpha=n_frames - 1).div_(
            n_frames
        )
        return current_scale_scores

    def compute_subset_relationships(
        self,
        current_scale_frame_idxs: torch.LongTensor,
        previous_scale_frame_idxs: torch.LongTensor,
    ) -> torch.BoolTensor:
        """

        Args:
            current_scale_frame_idxs: :math:`(N, F)`
            previous_scale_frame_idxs: :math:`(N', F-1)`

        Returns:
            :math:`(N, N')` bool tensor indicating subset relationships between
            current and previous scales

        """
        matching = torch.zeros(
            (current_scale_frame_idxs.shape[0], previous_scale_frame_idxs.shape[0]),
            dtype=torch.bool,
        )
        for i, current_set in enumerate(current_scale_frame_idxs.numpy()):
            current_set = set(current_set)
            for j, previous_set in enumerate(previous_scale_frame_idxs.numpy()):
                previous_set = set(previous_set)
                if previous_set.issubset(current_set):
                    matching[i, j] = True
        return cast(torch.BoolTensor, matching)

    def compute_shapley_contributions(
        self,
        current_scale_scores,
        current_scale_frame_idxs,
        previous_scale_scores,
        previous_scale_frame_idxs,
        n_video_frames,
        n_frames,
    ):
        if n_frames > 1:
            previous_scale_scores = torch.nn.functional.softmax(
                previous_scale_scores, dim=-1
            )
        current_scale_scores = torch.nn.functional.softmax(current_scale_scores, dim=-1)
        contributions = torch.zeros(
            (n_video_frames, self.n_classes), dtype=torch.float32, device=self.device
        )
        for frame_idx in range(n_video_frames):
            current_scale_with_i = (current_scale_frame_idxs == frame_idx).any(dim=-1)
            previous_scale_without_i = ~(
                (previous_scale_frame_idxs == frame_idx).any(dim=-1)
            )
            avg_score_with_i = current_scale_scores[current_scale_with_i].mean(dim=0)
            avg_score_without_i = previous_scale_scores[previous_scale_without_i].mean(
                dim=0
            )
            contributions[frame_idx] = avg_score_with_i - avg_score_without_i
        return contributions
