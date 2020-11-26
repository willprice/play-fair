from typing import Union, cast
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch.nn

from subset_samplers import ExhaustiveSubsetSampler
from subset_samplers import SubsetSampler
from tensor_ops import compute_subset_relations
from tensor_ops import masked_mean
from tensor_ops import masked_sum
from torch import nn


class OnlineExampleShapleyAttributor:
    """
    Debug info structure:
    self._debug_info["iter_<n>"][<scale_idx: [0, n_frames]>]:
        - subset_relations: bool tensor, (S_{scale_idx}, S_{scale_idx - 1})
          subset_relations[current_scale_subset_idx, previous_scale_subset_idx] ->
          whether the subset indexed by current_scale_subset_idx is a superset of
          the subset indexed by previous_scale_subset_idx.
        - frame_idx: long tensor, (S_{scale_idx}, {scale_idx + 1})
          frame_idx [subset_idx, position_idx] -> the frame_idx from the full sequence
          at position position_idx in the subset indexed by subset_idx.
        - scores_pre_ensemble: float tensor, (S_{scale_idx}, C)
          scores_pre_ensemble[subset_idx, class_idx] -> class score
        - scores_post_ensemble: float tensor, (S_{scale_idx}, C)
          scores_post_ensemble[subset_idx, class_idx] -> class score
        - current_softmax_scores: float tensor, (S_{scale_idx}, C)
          current_softmax_scores[current_scale_subset_idx, class_idx] -> class score
        - previous_softmax_scores: float tensor, (S_{scale_idx - 1}, C)
          previous_softmax_scores[previous_scale_subset_idx, class_idx] -> class score
        - current_scale_with_i: bool tensor, (n_frames, S_{scale_idx})
          current_scale_with_i[frame_idx, current_scale_subset_idx] -> whether
          the frame indexed by frame_idx is included in the subset indexed by
          current_scale_subset_idx.
        - previous_scale_without_i: bool tensor, (n_frames, S_{scale_idx - 1})
          previous_scale_without_i[frame_idx, previous_scale_subset_idx] -> whether
          the frame indexed by frame_idx is NOT included in the subset indexed by
          previous_scale_subset_idx.
        - summed_scores_with_i: float tensor, (n_frames,)
          summed_scores_with_i[frame_idx] -> summed class score
          for the class passed into `explain` over the subsets in the current
          scale containing the frame indexed by frame_idx.
        - summed_scores_without_i: float tensor, (n_frames,)
          summed_scores_without_i[frame_idx] -> summed class score
          for the class passed into `explain` over the subsets in the previous
          scale NOT containing the frame indexed by frame_idx.
        - n_summed_scores_with_i: long tensor, (n_frames,)
          n_summed_scores_with_i[frame_idx] -> number of subsets in the current scale
          containing the frame indexed by frame_idx.
        - n_summed_scores_without_i: long tensor, (n_frames,)
          n_summed_scores_without_i[frame_idx] -> number of subsets in the previous scale
          NOT containing the frame indexed by frame_idx.
    """

    def __init__(
        self,
        single_scale_models: Union[List[nn.Module], nn.ModuleList],
        priors: torch.Tensor,
        video: torch.Tensor,
        iterations: int,
        subset_sampler: SubsetSampler,
        device: torch.device,
        n_classes: int,
        debug: bool = False,
        count_n_evaluations: bool = True,
    ):
        self.single_scale_models = single_scale_models
        self.n_classes = n_classes

        assert priors.ndim == 2
        assert priors.shape[0] == 1
        self.priors = priors

        assert iterations >= 1
        self.n_iterations = iterations
        self.current_iteration = 0

        self.subset_sampler = subset_sampler
        self.device = device
        self.video = video
        self.n_video_frames = video.shape[0]
        self.debug = debug
        self._debug_info = dict()
        self.count_n_evaluations = count_n_evaluations

        self.summed_scores = torch.zeros(
            (
                iterations,
                2,
                self.n_video_frames + 1,
                self.n_video_frames,
                self.n_classes,
            ),
            dtype=torch.float64,
            device=self.device,
        )
        self.n_summed_scores = torch.zeros(
            (
                iterations,
                2,
                self.n_video_frames + 1,
                self.n_video_frames,
            ),
            dtype=torch.int64,
            device=self.device,
        )
        # (N', n_classes)
        self.previous_scores = self.priors.clone()
        self.current_scores = self.previous_scores

        # (1, 0)
        self.previous_frame_idxs = torch.tensor(
            [[]], dtype=torch.int64, device=self.device
        )
        self.previous_n_evaluations = torch.zeros(
            (1,), dtype=torch.float32, device=self.device
        )
        self.current_frame_idxs = self.previous_frame_idxs
        self.current_n_evaluations = self.previous_n_evaluations

        self.subset_relations = cast(
            torch.BoolTensor, torch.zeros((0, 0), dtype=torch.bool, device=self.device)
        )

    @property
    def _current_iteration_debug_info(self):
        iteration_key = f"iter_{self.current_iteration}"
        if iteration_key not in self._debug_info:
            self._debug_info[iteration_key] = dict()
        return self._debug_info[iteration_key]

    def run(self):
        for iteration in range(self.n_iterations):
            try:
                self.subset_sampler.reset()
            except AttributeError:
                pass
            self.current_iteration = iteration
            self.run_iter()

    def run_iter(self):
        for scale_index in range(0, self.n_video_frames):
            self._current_iteration_debug_info[scale_index] = scale_debug_info = dict()
            n_frames = scale_index + 1
            self.current_frame_idxs = self.subset_sampler.sample(
                self.n_video_frames, n_frames
            ).to(torch.long)
            self.subset_relations = compute_subset_relations(
                self.current_frame_idxs, self.previous_frame_idxs
            )
            self.current_n_evaluations = self.marginalise_n_evaluations() + 1

            if self.debug:
                scale_debug_info[
                    "subset_relations"
                ] = self.subset_relations.cpu().clone()
                scale_debug_info["frame_idx"] = self.current_frame_idxs.cpu().clone()

            if len(self.current_frame_idxs) == 0:
                continue

            # (T, C, ...) -> (N, n_frames, C, ...)
            subsequences = self.video[self.current_frame_idxs]

            if n_frames <= len(self.single_scale_models):
                single_scale_model = self.single_scale_models[scale_index]
                with torch.no_grad():
                    self.current_scores = single_scale_model(subsequences)
                if self.debug:
                    scale_debug_info[
                        "scores_pre_ensemble"
                    ] = self.current_scores.cpu().clone()
                if n_frames >= 2:
                    self.current_scores = self.combine_scores_under_nmax(n_frames)
            else:
                self.current_scores = self.combine_scores_over_nmax()
            if self.debug:
                scale_debug_info[
                    "scores_post_ensemble"
                ] = self.current_scores.cpu().clone()

            self.compute_shapley_contributions(scale_index)
            self.previous_scores = self.current_scores
            self.previous_frame_idxs = self.current_frame_idxs
            self.previous_n_evaluations = self.current_n_evaluations

    def compute_shapley_contributions(self, scale_index: int) -> None:
        device = self.current_scores.device
        # Don't softmax priors (they are already normalised to sum to 1!)
        if scale_index > 0:
            # (N', n_classes)
            previous_scale_scores = torch.nn.functional.softmax(
                self.previous_scores, dim=-1
            )
        else:
            previous_scale_scores = self.previous_scores

        # (N, n_classes)
        current_scale_scores = torch.nn.functional.softmax(self.current_scores, dim=-1)
        scale_debug_info = self._current_iteration_debug_info[scale_index]
        # (T, N)
        current_scale_with_i = cast(
            torch.BoolTensor,
            (
                self.current_frame_idxs[None, ...]
                == torch.arange(self.n_video_frames, device=device).reshape(-1, 1, 1)
            ).any(dim=-1),
        )
        # (T, N')
        previous_scale_without_i = cast(
            torch.BoolTensor,
            ~(
                (
                    self.previous_frame_idxs[None, ...]
                    == torch.arange(self.n_video_frames, device=device).reshape(
                        -1, 1, 1
                    )
                ).any(dim=-1)
            ),
        )
        if self.debug:
            scale_debug_info[
                "current_softmax_scores"
            ] = current_scale_scores.cpu().clone()
            scale_debug_info[
                "previous_softmax_scores"
            ] = previous_scale_scores.cpu().clone()
            scale_debug_info[
                "current_scale_with_i"
            ] = current_scale_with_i.cpu().clone()
            scale_debug_info[
                "previous_scale_without_i"
            ] = previous_scale_without_i.cpu().clone()
        # (T,)
        summed_scores_with_i = masked_sum(
            current_scale_with_i,
            current_scale_scores,
            bs=len(current_scale_with_i),
        )
        summed_scores_without_i = masked_sum(
            previous_scale_without_i,
            previous_scale_scores,
            bs=len(previous_scale_without_i),
        )
        n_summed_scores_with_i = current_scale_with_i.sum(dim=-1)
        n_summed_scores_without_i = previous_scale_without_i.sum(dim=-1)

        assert summed_scores_with_i.shape[0] == self.n_video_frames
        assert summed_scores_without_i.shape[0] == self.n_video_frames

        if self.debug:
            scale_debug_info["summed_scores_with_i"] = summed_scores_with_i.to(
                self.summed_scores.dtype
            )
            scale_debug_info["summed_scores_without_i"] = summed_scores_without_i.to(
                self.summed_scores.dtype
            )
            scale_debug_info["n_summed_scores_with_i"] = n_summed_scores_with_i
            scale_debug_info["n_summed_scores_without_i"] = n_summed_scores_without_i
        self.summed_scores[
            self.current_iteration, 0, scale_index + 1
        ] = summed_scores_with_i.to(self.summed_scores.dtype)
        self.summed_scores[
            self.current_iteration, 1, scale_index
        ] = summed_scores_without_i.to(self.summed_scores.dtype)
        self.n_summed_scores[
            self.current_iteration, 0, scale_index + 1
        ] = n_summed_scores_with_i
        self.n_summed_scores[
            self.current_iteration, 1, scale_index
        ] = n_summed_scores_without_i

    def combine_scores_under_nmax(
        self,
        n_frames: int,
        inplace: bool = True,
    ) -> torch.FloatTensor:
        if not inplace:
            self.current_scores = self.current_scores.clone()

        if self.count_n_evaluations:
            return self.current_scores.add_(
                other=masked_mean(
                    self.subset_relations,
                    self.previous_scores * self.previous_n_evaluations[:, None],
                )
            ).div_(self.current_n_evaluations[:, None])
        return self.current_scores.add_(
            alpha=n_frames - 1,
            other=masked_mean(self.subset_relations, self.previous_scores),
        ).div_(n_frames)

    def combine_scores_over_nmax(self) -> torch.FloatTensor:
        return cast(
            torch.FloatTensor, masked_mean(self.subset_relations, self.previous_scores)
        )

    def marginalise_n_evaluations(self) -> torch.FloatTensor:
        return cast(
            torch.FloatTensor,
            masked_mean(self.subset_relations, self.previous_n_evaluations),
        )


class OnlineShapleyAttributor:
    def __init__(
        self,
        single_scale_models: List[nn.Module],
        priors: np.ndarray,
        n_classes: int,
        device: Optional[torch.device] = None,
        subset_sampler: Optional[SubsetSampler] = None,
        debug: bool = False,
        count_n_evaluations: bool = True,
    ):
        self.single_scale_models = nn.ModuleList(single_scale_models).to(device).eval()
        self.priors = torch.from_numpy(priors).to(device).reshape(1, -1)
        self.n_classes = n_classes
        self.device = device
        if subset_sampler is None:
            subset_sampler = ExhaustiveSubsetSampler(device=self.device)
        self.subset_sampler = subset_sampler
        self.last_attributor = None
        self.debug = debug
        self.count_n_evaluations = count_n_evaluations

    def explain(
        self, video: torch.Tensor, n_iters: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the ESVs for a single video.

        Args:
            video: per-frame features of shape :math:`(T, D)` where
                :math:`D` is the dimensionality of the input feature.
            n_iters: How many times to run repeatedly run the joint model and ESV
                approximation, if using :py:class:`ExhaustiveSubsetSampler`, then there
                is no point setting this to anything but 1, this only makes a different
                when approximating ESVs.

        Returns:
            ESVs of shape :math:`(T, C)`
            where :math:`(t, c)` is the ESV for frame :math:`i` and class :math:`c`
        """
        attributor = OnlineExampleShapleyAttributor(
            single_scale_models=self.single_scale_models,
            priors=self.priors,
            video=video,
            iterations=n_iters,
            subset_sampler=self.subset_sampler,
            device=self.device,
            debug=self.debug,
            count_n_evaluations=self.count_n_evaluations,
            n_classes=self.n_classes,
        )
        self.last_attributor = attributor
        attributor.run()
        n_summed_scores = attributor.n_summed_scores.clone()
        summed_scores = attributor.summed_scores.sum(dim=0)
        n_summed_scores = n_summed_scores.sum(dim=0)
        n_summed_scores[n_summed_scores == 0] = 1
        avg_scores = (
            summed_scores
            / n_summed_scores.to(attributor.summed_scores.dtype)[..., None]
        )
        shapley_contributions = (avg_scores[0, 1:] - avg_scores[1, :-1]).to(
            torch.float32
        )
        shapley_values = shapley_contributions.mean(axis=0)
        return shapley_values, torch.softmax(
            attributor.current_scores.squeeze(0), dim=-1
        )
