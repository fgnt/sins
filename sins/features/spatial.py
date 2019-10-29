from __future__ import division, print_function

import numpy as np

from einops import rearrange
from math import ceil


class Coherence:
    def __init__(self, smooth_len=21):
        self.smooth_len = smooth_len

    def transform(self, x):
        psds = np.einsum('...ctf,...dtf->...tfcd', x, x.conj())

        if self.smooth_len is not None:
            pad_width = psds.ndim * [(0, 0)]
            pad_width[0] = (
                (self.smooth_len - 1) // 2, ceil((self.smooth_len - 1) / 2)
            )
            psds = np.pad(psds, pad_width=pad_width, mode='reflect')

            meanfilt = np.ones(self.smooth_len) / self.smooth_len
            psds = np.apply_along_axis(
                lambda m: np.correlate(m, meanfilt, mode='valid'),
                axis=0, arr=psds
            )

        apsds = np.diagonal(psds, axis1=-2, axis2=-1)
        coherence = psds / (
                np.sqrt(apsds[..., None] * apsds[..., None, :]) +
                np.finfo(psds.dtype).eps
        )
        triu_idx = np.triu(np.ones_like(coherence), k=1) > 0.5
        coherence = coherence[triu_idx].reshape(
            (*coherence.shape[:-2], -1)
        )
        angle = np.angle(coherence)
        coherence = np.concatenate(
            [np.abs(coherence), np.sin(angle), np.cos(angle)], axis=-1
        )
        assert np.all(np.abs(coherence)**2 >= 0.), np.min(np.abs(coherence)**2)
        assert (np.abs(coherence)**2 <= (1. + 1e-6)).all(), (np.abs(coherence)**2).max()
        return rearrange(coherence, 't f c -> c t f')

    def __call__(self, example):
        x = example['stft']
        example['coherence'] = self.transform(x)
        return example
