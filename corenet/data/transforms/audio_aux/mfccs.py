#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import math

import torch

from corenet.utils import logger


def get_mfccs(
    data: torch.Tensor,
    sampling_rate: float,
    num_mfccs: int,
    window_length: float = 0.023,
) -> torch.Tensor:
    """Get Mel Frequency Cepstral Coefficients from an audio signal.

    Explanation of Mel-Frequency Cepstral Coefficients (MFCCs):
    > https://librosa.org/doc/main/generated/librosa.stft.html#librosa.stft

    Args:
        data: one channel of the audio signal, as a 1-D tensor.
        sampling_rate: the sampling rate of the audio.
        num_mfccs: the number of cepstral coefficients to use.
        window_length: the window length used for computing the spectrogram. By
            default, we choose 23ms, which is a good value for human speech.
    """
    try:
        from torchaudio.transforms import (
            MFCC,  # Importing torchaudio takes ~0.6 s, but often not needed. That's why it is imported inside this function.
        )
    except ImportError:
        logger.error("Torchaudio is not installed. Please install it.")

    n_fft = sampling_rate * window_length
    # Round to the nearest power of 2.
    n_fft = 2 ** round(math.log2(n_fft))

    return MFCC(
        sample_rate=sampling_rate,
        n_mfcc=num_mfccs,
        melkwargs={
            "n_fft": n_fft,
            # librosa's default value: https://github.com/librosa/librosa/blob/71077174b9e73ae81d268f81551bb9667bf3693b/librosa/filters.py#L132
            "n_mels": 128,
            # librosa's default value: https://github.com/librosa/librosa/blob/71077174b9e73ae81d268f81551bb9667bf3693b/librosa/feature/spectral.py#L2027
            "hop_length": 512,
            "mel_scale": "slaney",
            # librosa's default value: https://github.com/librosa/librosa/blob/71077174b9e73ae81d268f81551bb9667bf3693b/librosa/filters.py#L136
            "norm": "slaney",
        },
    )(data.float())


def calculate_mfccs(
    audio: torch.Tensor,
    sampling_rate: float,
    num_mfccs: int,
    window_length: float = 0.023,
) -> torch.Tensor:
    """Calculate MFCCs on a batch of data.

    Args:
        audio: the audio signal, in [batch_size, num_channels, temporal_size]
            order.
        sampling_rate: the sampling rate of the audio signal.
        num_mfccs: the number of coefficients to use.
        window_length: the window length used for computing the spectrogram. By
            default, we choose 23ms, which is a good value for human speech.
    """
    if audio.dim() != 3:
        raise ValueError(f"Expected 3 dimensions, got {audio.dim()}")

    spectrogram_length = get_mfccs(
        audio[0][0],
        sampling_rate,
        num_mfccs,
        window_length=window_length,
    ).shape[1]

    result = torch.empty(
        [audio.shape[0], audio.shape[1], num_mfccs, spectrogram_length],
        device=audio.device,
        dtype=audio.dtype,
    )

    for i, element in enumerate(audio):
        for j, channel in enumerate(element):
            mfccs = get_mfccs(
                channel, sampling_rate, num_mfccs, window_length=window_length
            )
            result[i, j] = mfccs
    return result


def get_mfcc_features(
    audio: torch.Tensor,
    sampling_rate: float,
    num_mfccs: int,
    num_frames: int,
    window_length: float = 0.023,
) -> torch.Tensor:
    """Get MFCC features for a batch of audio data.

    Args:
        audio: the audio signal, in [batch_size, temporal_size, num_channels]
            order.
        sampling_rate: the sampling rate of the audio signal.
        num_mfccs: the number of coefficients to use.
        window_length: the window length used for computing the spectrogram. By
            default, we choose 23ms, which is a good value for human speech.
        num_frames: each MFCC spectrogram gets dividied into @num_frames frames
            (sub-time-slice temporal components) of length ceil(spectrogram_length/num_frames).
    Returns:
        MFCCs in [N, C, num_mfccs, num_frames, ceil(spectrogram_length/num_frames)] order.
    """
    if audio.dim() != 3:
        raise ValueError(f"Invalid audio.dim()={audio.dim()}")
    if audio.shape[2] != 2:
        raise ValueError(f"Invalid number of channels {audio.shape[2]}")
    audio = audio.permute([0, 2, 1])

    features = calculate_mfccs(
        audio,
        sampling_rate,
        num_mfccs,
        window_length=window_length,
    )  # Size: [N, C, num_mfccs, T].

    return get_padded_features(
        features=features,
        num_frames=num_frames,
    )


def get_padded_features(
    features: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """
    Splits the temporal dimension (of length T) of MFCC features into
    @num_frames sub-vectors (of length ``ceil(T/num_frames)``).
    As T may not be divisible by @num_frames, pads the temporal dimension if required.

    Args:
        features: Tensor[batchsize x C(num_audio_channels) x num_mfccs x T]
        num_frames: number of padded sub-vectors

    Returns:
        padded_features: Tensor (batchsize x C x num_mfccs x num_frames x ceil(T/num_frames))
    """
    N, C, num_mfccs, T = features.shape
    frame_length = math.ceil(T / num_frames)

    if T % num_frames != 0:
        padded_features = torch.zeros(
            [
                N,
                C,
                num_mfccs,
                frame_length * num_frames,
            ],
            dtype=features.dtype,
            device=features.device,
        )
        padded_features[:, :, :, :T] = features
    else:
        padded_features = features

    padded_features = padded_features.reshape(N, C, num_mfccs, num_frames, frame_length)
    return padded_features
