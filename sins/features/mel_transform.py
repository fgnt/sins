
import librosa
import numpy as np
from cached_property import cached_property


class MelTransform:
    def __init__(
            self,
            sample_rate: int,
            fft_length: int,
            n_mels: int = 64,
            fmin: int = 200,
            fmax: int = None,
            log: bool = True
    ):
        """
        Transforms stft to (log) mel spectrogram.

        Args:
            sample_rate: sample rate of audio signal
            fft_length: fft_length used in stft
            n_mels: number of filters to be applied
            fmin: lowest frequency (onset of first filter)
            fmax: highest frequency (offset of last filter)
        """
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.log = log

    @cached_property
    def fbanks(self):
        fbanks = librosa.filters.mel(
            n_mels=self.n_mels,
            n_fft=self.fft_length,
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=True,
            norm=None
        )
        fbanks = fbanks / fbanks.sum(axis=-1, keepdims=True)
        return fbanks.T

    def transform(self, stft):
        """

        Args:
            stft:

        Returns:

        """
        x = stft.real**2 + stft.imag**2
        x = np.dot(x, self.fbanks)
        if self.log:
            x = np.log(x + 1e-18)
        return x

    def __call__(self, example):
        stft = example["stft"]
        example["mel_transform"] = self.transform(stft)
        return example
