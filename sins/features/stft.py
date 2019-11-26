import numpy as np
from scipy.signal import stft


class STFT:
    def __init__(
            self,
            frame_step: int,
            fft_length: int,
            frame_length: int = None,
            window: str = "blackman",
            pad_mode: str = "mean"
    ):
        """transforms audio data to STFT.

        Args:
            frame_step:
            fft_length:
            frame_length:
            window:
            pad_mode:

        >>> stft = STFT(160, 512, 400)
        >>> audio_data=np.zeros((1,8000))
        >>> x = stft.transform(audio_data)
        >>> x.shape
        (1, 50, 257)
        """
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.frame_length = frame_length if frame_length is not None \
            else fft_length
        self.window = window
        self.pad_mode = pad_mode

    def transform(self, audio):
        if self.pad_mode is not None:
            pad = int((self.frame_length - self.frame_step) / 2)
            pad_width = (audio.ndim - 1) * [(0, 0)] + [
                (pad, pad + self.frame_step - 1)
            ]
            audio = np.pad(audio, pad_width=pad_width, mode=self.pad_mode)
        x = self.frame_length * stft(
            audio,
            nperseg=self.frame_length,
            noverlap=self.frame_length - self.frame_step,
            nfft=self.fft_length,
            window=self.window,
            axis=-1,
            padded=False,
            boundary=None
        )[-1]  # (..., F, T)
        x = np.moveaxis(x, -2, -1)
        return x

    def __call__(self, example):
        example["stft"] = self.transform(example["audio_data"])
        return example
