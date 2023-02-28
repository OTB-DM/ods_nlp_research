from typing import Iterable, List, Optional, Sequence, Union, Sized
from itertools import chain

import numpy as np


def is_str_batch(batch: Iterable) -> bool:
    """Checks if iterable argument contains string at any nesting level."""
    while True:
        if isinstance(batch, Iterable):
            if isinstance(batch, str):
                return True
            elif isinstance(batch, np.ndarray):
                return batch.dtype.kind == 'U'
            else:
                if len(batch) > 0:
                    batch = batch[0]
                else:
                    return True
        else:
            return False


def flatten_str_batch(batch: Union[str, Iterable]) -> Union[list, chain]:
    """Joins all strings from nested lists to one ``itertools.chain``.
    Args:
        batch: List with nested lists to flatten.
    Returns:
        Generator of flat List[str]. For str ``batch`` returns [``batch``].
    Examples:
        >>> [string for string in flatten_str_batch(['a', ['b'], [['c', 'd']]])]
        ['a', 'b', 'c', 'd']
    """
    if isinstance(batch, str):
        return [batch]
    else:
        return chain(*[flatten_str_batch(sample) for sample in batch])


def _get_all_dimensions(batch: Sequence, level: int = 0, res: Optional[List[List[int]]] = None) -> List[List[int]]:
    """Return all presented element sizes of each dimension.
    Args:
        batch: Data array.
        level: Recursion level.
        res: List containing element sizes of each dimension.
    Return:
        List, i-th element of which is list containing all presented sized of batch's i-th dimension.
    Examples:
        >>> x = [[[1], [2, 3]], [[4], [5, 6, 7], [8, 9]]]
        >>> _get_all_dimensions(x)
        [[2], [2, 3], [1, 2, 1, 3, 2]]
    """
    if not level:
        res = [[len(batch)]]
    if len(batch) and isinstance(batch[0], Sized) and not isinstance(batch[0], str):
        level += 1
        if len(res) <= level:
            res.append([])
        for item in batch:
            res[level].append(len(item))
            _get_all_dimensions(item, level, res)
    return res


def get_dimensions(batch: Sequence) -> List[int]:
    """Return maximal size of each batch dimension."""
    return list(map(max, _get_all_dimensions(batch)))


def zero_pad(batch: Sequence,
             zp_batch: Optional[np.ndarray] = None,
             dtype: type = np.float32,
             padding: Union[int, float] = 0) -> np.ndarray:
    """Fills the end of each array item to make its length maximal along each dimension.
    Args:
        batch: Initial array.
        zp_batch: Padded array.
        dtype = Type of padded array.
        padding = Number to will initial array with.
    Returns:
        Padded array.
    Examples:
        >>> x = np.array([[1, 2, 3], [4], [5, 6]])
        >>> zero_pad(x)
        array([[1., 2., 3.],
               [4., 0., 0.],
               [5., 6., 0.]], dtype=float32)
    """
    if zp_batch is None:
        dims = get_dimensions(batch)
        zp_batch = np.ones(dims, dtype=dtype) * padding
    if zp_batch.ndim == 1:
        zp_batch[:len(batch)] = batch
    else:
        for b, zp in zip(batch, zp_batch):
            zero_pad(b, zp)
    return zp_batch

