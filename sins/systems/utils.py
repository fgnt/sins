
import numpy as np
import torch

from sins.utils import nested_op


class Collate:
    """

    >>> batch = [{'a': np.ones((5,2)), 'b': '0'}, {'a': np.ones((3,2)), 'b': '1'}]
    >>> Collate()(batch)
    {'a': tensor([[[1., 1.],
             [1., 1.],
             [1., 1.],
             [1., 1.],
             [1., 1.]],
    <BLANKLINE>
            [[1., 1.],
             [1., 1.],
             [1., 1.],
             [0., 0.],
             [0., 0.]]]), 'b': ['0', '1']}
    """
    def __init__(self, to_tensor=True):
        """
        Collates a list of example dicts to a dict of lists, where lists of
        numpy arrays are stacked. Optionally casts numpy arrays to torch Tensors.

        Args:
            to_tensor:
        """
        self.to_tensor = to_tensor

    def __call__(self, example, training=False):
        example = nested_op(self.collate, *example, sequence_type=())
        return example

    def collate(self, *batch):
        batch = list(batch)
        if isinstance(batch[0], np.ndarray):
            max_len = np.zeros_like(batch[0].shape)
            for array in batch:
                max_len = np.maximum(max_len, array.shape)
            for i, array in enumerate(batch):
                pad = max_len - array.shape
                if np.any(pad):
                    assert np.sum(pad) == np.max(pad), (
                        'arrays are only allowed to differ in one dim',
                    )
                    pad = [(0, n) for n in pad]
                    batch[i] = np.pad(array, pad_width=pad, mode='constant')
            batch = np.array(batch).astype(batch[0].dtype)
            if self.to_tensor:
                import torch
                batch = torch.from_numpy(batch)
        return batch


def batch_to_device(batch, device=None):
    """
    Moves a nested structure to the device.
    Numpy arrays are converted to torch.Tensor, except complex numpy arrays
    that aren't supported in the moment in torch.

    The original doctext from torch for `.to`:
    Tensor.to(device=None, dtype=None, non_blocking=False, copy=False) → Tensor
        Returns a Tensor with the specified device and (optional) dtype. If
        dtype is None it is inferred to be self.dtype. When non_blocking, tries
        to convert asynchronously with respect to the host if possible, e.g.,
        converting a CPU Tensor with pinned memory to a CUDA Tensor. When copy
        is set, a new Tensor is created even when the Tensor already matches
        the desired conversion.

    Args:
        batch:
        device: None, 'cpu', 0, 1, ...

    Returns:
        batch on device

    """

    if isinstance(batch, dict):
        return batch.__class__({
            key: batch_to_device(value, device=device)
            for key, value in batch.items()
        })
    elif isinstance(batch, (tuple, list)):
        return batch.__class__([
            batch_to_device(element, device=device)
            for element in batch
        ])
    elif torch.is_tensor(batch):
        return batch.to(device=device)
    elif isinstance(batch, np.ndarray):
        if batch.dtype in [np.complex64, np.complex128]:
            # complex is not supported
            return batch
        else:
            # TODO: Do we need to ensure tensor.is_contiguous()?
            # TODO: If not, the representer of the tensor does not work.
            return batch_to_device(
                torch.from_numpy(batch), device=device
            )
    elif hasattr(batch, '__dataclass_fields__'):
        return batch.__class__(
            **{
                f: batch_to_device(getattr(batch, f), device=device)
                for f in batch.__dataclass_fields__
            }
        )
    else:
        return batch
