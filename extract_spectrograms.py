#!usr/bin/env python
#-*- coding: utf8 -*-

# Code borrowed from David Harwath
# https://github.com/dharwath/DAVEnet-pytorch
#
#   BibleNet
#
#   Modification by: GETALP TEAM
#   Last Modified: 27/03/2019
#
#   Université Grenoble Alpes

import librosa
import numpy as np
import sys

def preemphasis(signal, coeff=0.97):
    """
    perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def extract_spectogram(file_path):
    preemph_coef = 0.97
    sample_rate = 16000
    window_size = 0.025
    window_stride = 0.01
    window_type = 'hamming'
    num_mel_bins = 40
    fmin = 20
    n_fft = int(sample_rate * window_size)
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    # load audio, subtract DC, preemphasis
    y, sr = librosa.load(file_path, sample_rate)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)

    # compute mel spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_type)
    spec = np.abs(stft) ** 2
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
    melspec = np.dot(mel_basis, spec)
    logspec = librosa.power_to_db(melspec, ref=np.max)

    logspec = logspec.T
    return logspec

def main(alignment_table_path):
    alignment_table = np.genfromtxt(alignment_table_path, delimiter=',', invalid_raise=False, dtype=str)

    for i_language in range(2, len(alignment_table[0])):
        spectrograms = []
        language_name = alignment_table[0,i_language]
        wav_base_path = alignment_table[1,i_language]
        wav_true_path = '{}{}_one_channel.wav'
        print(language_name)
        for line in range(2, len(alignment_table)):
            verse_id = alignment_table[line, 0]
            verse_lang_name = alignment_table[line, i_language]
            print('\t{} {}'.format(verse_id, verse_lang_name))
            if verse_lang_name != 'Not Available':
                mel_spec = extract_spectogram(wav_true_path.format(wav_base_path, verse_lang_name))
            else:
                mel_spec = []
            spectrograms.append(mel_spec)
        np.save('{}_mel_spec.npy'.format(language_name), spectrograms)
        # print(alignment_table[2,i_language])



if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0])