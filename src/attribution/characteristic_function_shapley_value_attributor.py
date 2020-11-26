from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from subset_samplers import ExhaustiveSubsetSampler
from subset_samplers import SubsetSampler
from tensor_ops import masked_sum


class CharacteristicFunctionShapleyAttributor:
    def __init__(
        self,
        characteristic_fn: Callable[[torch.Tensor], torch.Tensor],
        n_classes: int,
        subset_sampler: Optional[SubsetSampler] = None,
        device: torch.device = None,
    ):
        self.characteristic_fn = characteristic_fn
        self.n_classes = n_classes
        if subset_sampler is None:
            subset_sampler = ExhaustiveSubsetSampler(device=device)
        self.subset_sampler = subset_sampler
        self.device = device
        self.last_attributor = None

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
            where :math:`(t, c)` is the ESV for frame :math:`i` and class :math:`c`.
        """
        self.last_attributor = (
            attributor
        ) = CharacteristicFunctionExampleShapleyAttributor(
            video,
            characteristic_fn=self.characteristic_fn,
            n_classes=self.n_classes,
            subset_sampler=self.subset_sampler,
            device=self.device,
            iterations=n_iters
        )
        return attributor.run()


class CharacteristicFunctionExampleShapleyAttributor:
    def __init__(
        self,
        video: torch.Tensor,
        characteristic_fn: Callable[[torch.Tensor], torch.Tensor],
        iterations: int,
        n_classes: int,
        subset_sampler: Optional[SubsetSampler] = None,
        device: torch.device = None,
        characteristic_fn_args: Optional[Tuple[Any]] = None,
        characteristic_fn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if characteristic_fn_args is None:
            characteristic_fn_args = []

        if characteristic_fn_kwargs is None:
            characteristic_fn_kwargs = {}

        self.characteristic_fn = characteristic_fn
        self.characteristic_fn_args = characteristic_fn_args
        self.characteristic_fn_kwargs = characteristic_fn_kwargs
        self.sequence_features = video
        self.n_classes = n_classes
        self.n_elements = len(video)
        self.n_iterations = iterations
        if subset_sampler is None:
            subset_sampler = ExhaustiveSubsetSampler(device=device)
        self.subset_sampler = subset_sampler
        self.device = device
        self.n_scales = self.n_elements + 1
        self.summed_scores = torch.zeros(
            (self.n_iterations, 2, self.n_scales, self.n_elements, self.n_classes),
            device=device,
            dtype=torch.float32,
        )
        self.n_summed_scores = torch.zeros(
            (self.n_iterations, 2, self.n_scales, self.n_elements),
            device=device,
            dtype=torch.long,
        )

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for iteration in range(self.n_iterations):
            try:
                self.subset_sampler.reset()
            except AttributeError:
                pass
            self.run_iter(iteration)
        attributions = self._compute_attributions()
        grand_coalition_scores = self.characteristic_fn(
            self.sequence_features.unsqueeze(0)
        ).squeeze(0)
        return attributions, grand_coalition_scores

    def run_iter(self, iter_idx: int):
        for scale_index in range(self.n_scales):
            n_subset_elements = scale_index
            # (N, T)
            subset_idxs = self.subset_sampler.sample(
                self.n_elements, n_subset_elements
            ).to(torch.long)
            # (N, C)
            scores = self.characteristic_fn(self.sequence_features[subset_idxs])
            # (T, N)
            with_i = cast(
                torch.BoolTensor,
                (
                    subset_idxs[None, :, :]
                    == torch.arange(self.n_elements, device=self.device)[:, None, None]
                ).any(dim=-1),
            )
            without_i = ~with_i
            # (T, N)
            summed_scores_with_i = masked_sum(with_i, scores)
            summed_scores_without_i = masked_sum(without_i, scores)
            self.summed_scores[iter_idx, 0, scale_index] = summed_scores_with_i
            self.summed_scores[iter_idx, 1, scale_index] = summed_scores_without_i
            self.n_summed_scores[iter_idx, 0, scale_index] = with_i.sum(dim=-1)
            self.n_summed_scores[iter_idx, 1, scale_index] = without_i.sum(dim=-1)

    def _compute_attributions(self):
        # Sum up over iterations
        # [with/without_i index, scale index, element index, class index] -> summed
        # scores with element at scale
        summed_scores = self.summed_scores.sum(dim=0)
        # [with/without_i index, scale index, element index, class index] -> number of
        # results with element at scale
        n_summed_scores = self.n_summed_scores.sum(dim=0)
        no_results_bool_idx = n_summed_scores == 0
        n_summed_scores[no_results_bool_idx] = 1
        avg_scores = summed_scores / n_summed_scores[..., None].to(torch.float32)
        # [scale index, element index, class index] -> expected marginal contribution
        marginal_contributions = avg_scores[0, 1:] - avg_scores[1, :-1]
        shapley_values = marginal_contributions.mean(0)
        return shapley_values
