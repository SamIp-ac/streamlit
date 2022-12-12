import os

import numpy as np
import soundfile as sf

import matplotlib.pyplot as plt

from ic_model import params as yamnet_params
from ic_model import yamnet as yamnet_model
import tensorflow as tf


def inst_classifier(filename):
    # Read in the audio.
    wav_file_name = filename
    wav_data_raw, sr = sf.read(wav_file_name, dtype=np.int16, always_2d=False)

    try:
        if wav_data_raw.shape[1] == 1:
            wav_data = wav_data_raw
        elif wav_data_raw.shape[1] == 2:
            wav_data = (wav_data_raw[:, 0] + wav_data_raw[:, 1]) / 2
    except:
        wav_data = wav_data_raw

    # print(max(wav_data))
    waveform = wav_data / 32768.0
    # print(max(waveform))

    # The graph is designed for a sampling rate of 16 kHz, but higher rates should work too.
    # We also generate scores at a 10 Hz frame rate.
    params = yamnet_params.Params(sample_rate=sr, patch_hop_seconds=0.1)

    # Set up the YAMNet model.
    class_names = yamnet_model.class_names('ic_model/yamnet_class_map.csv')
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('ic_model/yamnet.h5')

    # Run the model.
    scores, embeddings, spectrogram = yamnet(waveform)
    scores = scores.numpy()
    spectrogram = spectrogram.numpy()

    # Visualize the results.
    plt.figure(figsize=(10, 8))

    # Plot the waveform.
    plt.subplot(3, 1, 1)
    plt.plot(waveform)
    plt.xlim([0, len(waveform)])
    # Plot the log-mel spectrogram (returned by the model).
    plt.subplot(3, 1, 2)
    plt.imshow(spectrogram.T, aspect='auto', interpolation='nearest', origin='lower')

    # Plot and label the model output scores for the top-scoring classes.
    mean_scores = np.mean(scores, axis=0)
    top_N = 10
    top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
    plt.subplot(3, 1, 3)
    plt.imshow(scores[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
    # Compensate for the patch_window_seconds (0.96s) context window to align with spectrogram.
    patch_padding = (params.patch_window_seconds / 2) / params.patch_hop_seconds
    plt.xlim([-patch_padding, scores.shape[0] + patch_padding])
    # Label the top_N classes.
    yticks = range(0, top_N, 1)
    plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
    _ = plt.ylim(-0.5 + np.array([top_N, 0]))

    plt.savefig('pages/Data/temp.png')
    top_ = [class_names[top_class_indices[x]] for x in range(0, 5, 1)]
    return top_
