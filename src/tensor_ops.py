import itertools as it
import random

from typing import cast
from typing import Optional

import numpy as np
import torch

from scipy.special import comb


def broadcast_cat(tensors, *args, **kwargs):
    return torch.cat(torch.broadcast_tensors(*tensors), *args, **kwargs)


def sum_all_but(tensor: torch.Tensor, index: int, dim: int = -1) -> torch.Tensor:
    """

    Args:
        tensor: Tensor of shape `(N_1, ..., N_j, ..., N_n)`
        index: Element index to preserve in dimension :math:`j`
        dim: Dimension to index (:math:`j` in ``tensor`` shape) between :math:`0` and :math`n - 1`

    Returns:
        Tensor of shape `(N_1, ..., 2, ..., N_n)` where the original :math:`N_j`
        elements were collapsed:
        - ``out[N_1, ..., 0, ..., N_n] = tensor[N_1, ..., index, ..., N_n]``
        - ``out[N_1, ..., 1, ..., N_n] = tensor[N_1, ..., ~index, ..., N_n]``.sum(dim=j)
        where ``~index`` represents every other element in dimension ``j`` that isn't
        ``index``.

    """
    max_index = tensor.shape[dim]
    assert 0 <= index <= max_index, f"Expected 0 <= {index} <= {max_index}"
    ndim = tensor.ndimension()
    if dim < 0:
        dim = ndim + dim
    leading_indices = [slice(None) for _ in range(dim)]
    trailing_indices = [slice(None) for _ in range(ndim - dim - 1)]

    return torch.stack(
        [
            tensor[(*leading_indices, index, *trailing_indices)],
            tensor[(*leading_indices, slice(0, index), *trailing_indices)].sum(dim=dim)
            + tensor[(*leading_indices, slice(index + 1, None), *trailing_indices)].sum(
                dim=dim
            ),
        ],
        dim=dim,
    )


def subsequence_indices(
    sequence_length: int, subsequence_length: int, device=None
) -> torch.Tensor:
    """Generate indices for every possible subsequence of length K from a sequence of
    length N.

    Args:
        sequence_length: :math:`N`
        subsequence_length: :math:`K`

    Returns:
        Tensor of shape :math:`({N \\choose K}, K)` where each row contains
        :math:`K` indices between :math:`0` and :math:`N - 1`.
    """
    # WARNING: Do NOT use `torch.combinations` for this implementation
    # it is crazy slow and uses so much memory. The itertools/np.fromiter implementation
    # is much faster.
    # This implementation is from SO:
    # https://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy

    from scipy.special import comb

    if subsequence_length == 0:
        return torch.zeros((1, 0), dtype=torch.long, device=device)
    count = comb(sequence_length, subsequence_length, exact=True)
    subsequence_idx = np.fromiter(
        it.chain.from_iterable(
            it.combinations(range(sequence_length), subsequence_length)
        ),
        np.int64,
        count=count * subsequence_length,
    ).reshape(-1, subsequence_length)
    return torch.from_numpy(subsequence_idx).to(device)


def random_subsequence_indices(
    sequence_length: int,
    subsequence_length: int,
    sample_size: int = 1,
    exclude_repeats: bool = False,
    device=None,
) -> torch.Tensor:
    """Generate L randomly sampled ordered indices for subsequences of length K from a
    sequence of length N.

    Args:
        sequence_length: :math:`N`
        subsequence_length: :math:`K`
        sample_size: :math:`L`

    Returns:
        Tensor of shape :math:`(L, K)` where each row contains
        :math:`K` indices between :math:`0` and :math:`N - 1`.
    """
    if subsequence_length == 0:
        return torch.zeros((1, 0), dtype=torch.long, device=device)

    population = list(range(sequence_length))
    if exclude_repeats:
        sample_size = min(
            comb(sequence_length, subsequence_length, exact=True), sample_size
        )
        samples = set()
        while len(samples) != sample_size:
            samples.add(frozenset(random.sample(population, k=subsequence_length)))
        return torch.from_numpy(np.stack([sorted(s) for s in samples])).to(device)

    return (
        torch.from_numpy(
            np.sort(
                np.stack(
                    [
                        np.random.choice(
                            population, size=subsequence_length, replace=False
                        )
                        for _ in range(sample_size)
                    ]
                ),
                axis=-1,
            )
        )
        .to(device)
        .to(torch.long)
    )


def masked_sum(
    mask: torch.BoolTensor,
    input: torch.Tensor,
    bs: int = 100,
    nan_replacement: Optional[bool] = False,
):
    """Compute sum of masked elements of ``input``.

    Args:
        mask: :math:`(N, M)`
        input: :math:`(M, ...)`
        bs: Block size used to break up iteration, decrease this number if running
            into OOM errors.
        nan_replacement: Substitute mean with NaN when mask row is all False

    Returns:
        :math:`(N, ...)`
    """
    assert bs >= 1
    n_batches = int(np.ceil(len(mask) / bs))
    block_results = []
    trailing_dims = [None] * (len(input.shape) - 1)
    for i in range(n_batches):
        # (bs, M, 1...)
        broadcast_mask = mask[(slice(i * bs, (i + 1) * bs), ..., *trailing_dims)].to(
            input.dtype
        )
        # (1, M, ...)
        broadcast_input = input[None, ...]
        # (bs, ...)
        sum = (broadcast_mask * broadcast_input).sum(dim=1)
        if nan_replacement:
            n_selected_elements = broadcast_mask.sum(dim=1)
            sum.masked_fill_(n_selected_elements == 0, np.nan)
        block_results.append(sum)
    out = torch.cat(block_results, dim=0)
    return out


def masked_mean(
    mask: torch.BoolTensor,
    input: torch.Tensor,
    bs: int = 100,
    nan_replacement: Optional[bool] = False,
) -> torch.Tensor:
    """Compute mean of masked elements of ``input``.

    Args:
        mask: :math:`(N, M)`
        input: :math:`(M, ...)`
        bs: Block size used to break up iteration, decrease this number if running
            into OOM errors.
        nan_replacement: Substitute mean with NaN when mask row is all False

    Returns:
        :math:`(N, ...)`
    """
    assert mask.dtype == torch.bool

    assert bs >= 1
    n_batches = int(np.ceil(len(mask) / bs))
    block_results = []
    trailing_dims = [None] * (len(input.shape) - 1)

    for i in range(n_batches):
        # (bs, M, 1...)
        broadcast_mask = mask[(slice(i * bs, (i + 1) * bs), ..., *trailing_dims)].to(
            input.dtype
        )
        # (1, M, ...)
        broadcast_input = input[None, ...]
        # (bs, 1...)
        n_selected_elements = broadcast_mask.sum(dim=1)
        # (bs, ...)
        sum = (broadcast_mask * broadcast_input).sum(dim=1)
        zero_idx = n_selected_elements == 0
        if nan_replacement:
            sum.masked_fill_(zero_idx, np.nan)

        # Prevent divide by 0 errors.
        n_selected_elements.masked_fill_(zero_idx, 1)
        block_results.append(sum / n_selected_elements)
    out = torch.cat(block_results, dim=0)
    return out


def random_sample_elements(ls: torch.Tensor, n: int, replace=False) -> torch.Tensor:
    if len(ls) > n:
        idxs = np.sort(np.random.choice(np.arange(len(ls)), size=n, replace=replace))
        return ls[torch.from_numpy(idxs)]
    return ls


def compute_subset_relations(
    current_scale_frame_idx: torch.Tensor,
    previous_scale_frame_idx: torch.Tensor,
    bs: Optional[int] = None,
) -> torch.BoolTensor:
    """
    Args:
        current_scale_frame_idx: :math:`(N, S)`
        previous_scale_frame_idx:  :math:`(N', S-1)`
        bs: Block size, used to divide up computation to reduce memory burden.

    Returns:
        Bool tensor of shape :math:`(N, N')` where ``out[i, j]`` indicates where
        ``previous_scale_frame_idx[j]`` is a subset of
        ``current_scale_frame_idx[i]``.
    """
    assert current_scale_frame_idx.ndimension() == 2
    assert previous_scale_frame_idx.ndimension() == 2
    if bs is None:
        if (
            np.prod(current_scale_frame_idx.shape)
            * np.prod(previous_scale_frame_idx.shape)
        ) >= 1e7:
            # this was empirically determined to be about as high as we can get
            # away with without blowing up to more than 10GB of GPU mem.
            bs = int(
                1e7
                / max(
                    current_scale_frame_idx.shape[0], previous_scale_frame_idx.shape[0],
                )
            )
        else:
            bs = current_scale_frame_idx.shape[0]

    n_batches = int(np.ceil(current_scale_frame_idx.shape[0] / bs))
    # (bs, N', S-1, S)
    batched_comparison = []
    for i in range(n_batches):
        batch_idx = slice(i * bs, (i + 1) * bs)
        comp = (
            current_scale_frame_idx[batch_idx, None, None, :]
            == previous_scale_frame_idx[None, :, :, None]
        )
        batched_comparison.append(comp.any(dim=-1).all(dim=-1))
    # (N, N')
    subset_children = torch.cat(batched_comparison, dim=0)
    assert subset_children.shape[0] == current_scale_frame_idx.shape[0]
    return cast(torch.BoolTensor, subset_children)
