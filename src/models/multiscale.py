from collections import defaultdict
from typing import List, Union
from typing import Optional

import torch
import torch.nn.functional as F

from subset_samplers import ExhaustiveSubsetSampler
from subset_samplers import SubsetSampler
from tensor_ops import compute_subset_relations
from tensor_ops import masked_mean
from torch import nn


class MultiscaleModel(nn.Module):
    r"""
    Implements the multiscale model defined as:

    .. math::

        f(X) = \mathbb{E}_s \left[
            \mathbb{E}_{\substack{X' \subseteq X \\ |X'| = s}} [f_s(X')]
        \right]
    """

    def __init__(
        self,
        single_scale_models: Union[List[nn.Module], nn.ModuleList],
        softmax: bool = False,
        save_intermediate: bool = False,
        sampler: Optional[SubsetSampler] = None,
    ):
        super().__init__()
        self.single_scale_models = nn.ModuleList([model for model in single_scale_models])

        if sampler is None:
            self.sampler = ExhaustiveSubsetSampler()
        else:
            self.sampler = sampler
        self.softmax = softmax
        self.save_intermediate = save_intermediate
        self.intermediates = None

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: Example of shape :math:`(T, C)`  where :math:`T` is the number of
            elements in the sequence and :math:`C` is the channel size.
        Returns:
            Class scores of shape `(C',)` where :math:`C'` is the number of classes.
        """
        try:
            self.sampler.reset()
        except AttributeError:
            pass
        if self.save_intermediate:
            self.intermediates = defaultdict(lambda: dict())
        sequence_len = sequence.shape[0]
        scores = None
        n_scales = min(sequence_len, len(self.single_scale_models))
        for scale_idx in range(n_scales):
            subsequence_len = scale_idx + 1
            current_subsequence_idxs = self.sampler.sample(
                sequence_len, subsequence_len
            )
            subsequences = sequence[current_subsequence_idxs]
            if scale_idx < len(self.single_scale_models):
                single_scale_model = self.single_scale_models[scale_idx]
                current_scores = single_scale_model(subsequences)
                if self.softmax:
                    current_scores = F.softmax(current_scores, dim=-1)
            if scores is None:
                scores = current_scores.mean(dim=0)
            else:
                scores.add_(current_scores.mean(dim=0))

            if self.save_intermediate:
                self.intermediates[scale_idx]["ensembled_scores"] = scores.cpu().numpy()
                self.intermediates[scale_idx]["scores"] = current_scores.cpu().numpy()
                self.intermediates[scale_idx][
                    "subsequence_idxs"
                ] = current_subsequence_idxs
        scores.div_(n_scales)

        return scores

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
                *args, **kwargs
        )
        self.sampler.device = device
        return super().to(*args, **kwargs)


class RecursiveMultiscaleModel(nn.Module):
    r"""
    Implements the multiscale model defined as:

    .. math::

        f(X) = \mathbb{E}_s \left[
            \mathbb{E}_{\substack{X' \subseteq X \\ |X'| = s}} [f_s(X')]
        \right]

    But rather than computing it in this fashion, it computes it in a bottom fashion
    so that :math:`f(X')` for subsets :math:`X'` is computed and used to produce the
    output for the next scale up. This way we get the outputs `f(X')` as a side effect
    of computing `f(X)` for free. This is useful for Shapely Value analysis where we
    need these intermediate values. This is accomplished by reformulating the above
    into a recurrence:

    .. math::

        f(X) = \begin{cases}
            \mathbb{E}_{\substack{X' \subseteq X \\ |X'| = |X| - 1}}[f(X')]
            & |X| \geq n_{\max{}} \\

            |X|^{-1} (
                f_{|X|}(X) +
                (|X| - 1) \mathbb{E}_{\substack{X' \subset X \\ |X'| = |X| - 1}}[f( X')]
            & \text{otherwise}
        \end{cases}
    """

    def __init__(
        self,
        single_scale_models: Union[List[nn.Module], nn.ModuleList],
        save_intermediate: bool = False,
        sampler: Optional[SubsetSampler] = None,
        count_n_evaluations: bool = False,
        softmax: bool = False,
    ):
        super().__init__()
        self.single_scale_models = nn.ModuleList([model for model in single_scale_models])
        self.softmax = softmax
        if sampler is None:
            self.sampler = ExhaustiveSubsetSampler()
        else:
            self.sampler = sampler
        self.save_intermediate = save_intermediate
        self.intermediates = None
        self.count_n_evaluations = count_n_evaluations

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: Example of shape :math:`(T, C)`  where :math:`T` is the number of
            elements in the sequence and :math:`C` is the channel size.
        Returns:
            Class scores of shape `(C',)` where :math:`C'` is the number of classes.
        """
        if self.save_intermediate:
            self.intermediates = defaultdict(lambda: dict())
        sequence_len = sequence.shape[0]
        previous_scores = torch.zeros(
            (0, self.single_scale_models[0].model[-1].out_features),
            dtype=torch.float,
            device=sequence.device,
        )
        previous_subsequence_idx = torch.tensor(
            [[]], dtype=torch.long, device=sequence.device
        )
        previous_n_evaluations = torch.zeros(
            (1,), dtype=torch.float32, device=sequence.device
        )

        try:
            self.sampler.reset()
        except AttributeError:
            pass
        for scale_idx in range(sequence_len):
            subsequence_len = scale_idx + 1
            current_subsequence_idxs = self.sampler.sample(
                sequence_len, subsequence_len
            )
            subsequences = sequence[current_subsequence_idxs]
            subset_relations = compute_subset_relations(
                current_subsequence_idxs, previous_subsequence_idx
            )

            current_n_evaluations = (
                masked_mean(subset_relations, previous_n_evaluations)
            ) + 1
            if scale_idx < len(self.single_scale_models):
                single_scale_model = self.single_scale_models[scale_idx]
                current_scores = single_scale_model(subsequences)
                if self.softmax:
                    current_scores = F.softmax(current_scores, dim=-1)
            else:
                current_scores = masked_mean(subset_relations, previous_scores)

            if self.save_intermediate:
                self.intermediates[scale_idx]["scores"] = current_scores.cpu().numpy()

            if sequence_len >= 2 and 0 < scale_idx < len(self.single_scale_models):
                if self.count_n_evaluations:
                    current_scores.add_(
                        other=masked_mean(
                            subset_relations,
                            previous_scores * previous_n_evaluations[:, None],
                        )
                    ).div_(current_n_evaluations[:, None])
                else:
                    current_scores.add_(
                        alpha=scale_idx,
                        other=masked_mean(subset_relations, previous_scores),
                    ).div_(scale_idx + 1)

            if self.save_intermediate:
                self.intermediates[scale_idx][
                    "ensembled_scores"
                ] = current_scores.cpu().numpy()
                self.intermediates[scale_idx][
                    "subsequence_idxs"
                ] = current_subsequence_idxs.cpu().numpy()
                self.intermediates[scale_idx][
                    "current_n_evaluations"
                ] = current_n_evaluations.cpu().numpy()

            previous_scores = current_scores
            previous_subsequence_idx = current_subsequence_idxs
            previous_n_evaluations = current_n_evaluations

        return current_scores.mean(dim=0)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        self.sampler.device = device
        return super().to(*args, **kwargs)
