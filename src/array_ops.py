import numpy as np


def select(xs: np.ndarray, xs_ids: np.ndarray, selection_ids: np.ndarray) -> np.ndarray:
    """

    Args:
        xs: Array to select elements from
        xs_ids: Array of ids for each element in xs
        selection_ids: Array of ids to select

    Returns:
        A selection of elements from ``xs``

    Examples:

        >>> select(\
            np.array([1, 2, 3]), \
            np.array(['a', 'b', 'c']), \
            np.array(['a']) \
        )
        array([1])
        >>> select(\
            np.array([1, 2, 3]), \
            np.array(['a', 'b', 'c']), \
            np.array(['a', 'c']) \
        )
        array([1, 3])
        >>> select(\
            np.array([1, 2, 3]), \
            np.array(['a', 'b', 'c']), \
            np.array(['a', 'c', 'a']) \
        )
        array([1, 3, 1])

    """
    if len(xs) != len(xs_ids):
        raise ValueError(
                f"Expected xs and xs_ids to be the same length but were "
                f"{len(xs)} and {len(xs_ids)} respectively."
        )
    xs_lookup = {id: idx for idx, id in enumerate(xs_ids)}
    chosen_idx = np.array([xs_lookup[id] for id in selection_ids], dtype=np.int)
    return xs[chosen_idx]

