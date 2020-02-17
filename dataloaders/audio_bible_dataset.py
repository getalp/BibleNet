#!usr/bin/env python
#-*- coding: utf8 -*-

# Author of Original Implementation: David Harwath
# https://github.com/dharwath/DAVEnet-pytorch
#
#   BibleNet
#
#   Modification by: GETALP TEAM
#   Last Modified: 27/03/2019
#
#   Université Grenoble Alpes

import json
import librosa
import numpy as np
import os
from PIL import Image
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def preemphasis(signal,coeff=0.97):
    """
    perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])


class AudioBibleDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf=None):
        """
        Dataset that manages a set of paired images and audio recordings

        :param dataset_json_file
        :param audio_conf: Dictionary containing the sample rate, window and
        the window length/stride in seconds, and normalization to perform (optional)
        :param image_transform: torchvision transform to apply to the images (optional)
        """
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_langA_base_path = data_json['audio_langA_base_path']
        self.audio_langB_base_path = data_json['audio_langB_base_path']

        if not audio_conf:
            self.audio_conf = {}
        else:
            self.audio_conf = audio_conf

        self.windows = {'hamming': scipy.signal.hamming,
                        'hann': scipy.signal.hann, 
                        'blackman': scipy.signal.blackman,
                        'bartlett': scipy.signal.bartlett
                        }

    def _LoadAudio(self, path):
        audio_type = self.audio_conf.get('audio_type', 'melspectrogram')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        preemph_coef = self.audio_conf.get('preemph_coef', 0.97)
        sample_rate = self.audio_conf.get('sample_rate', 16000)
        window_size = self.audio_conf.get('window_size', 0.025)
        window_stride = self.audio_conf.get('window_stride', 0.01)
        window_type = self.audio_conf.get('window_type', 'hamming')
        num_mel_bins = self.audio_conf.get('num_mel_bins', 40)
        target_length = self.audio_conf.get('target_length', 1024)
        use_raw_length = self.audio_conf.get('use_raw_length', True)
        padval = self.audio_conf.get('padval', 0)
        fmin = self.audio_conf.get('fmin', 20)
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        y, sr = librosa.load(path, sample_rate)
        if y.size == 0:
            y = np.zeros(200)
        y = y - y.mean()
        y = preemphasis(y, preemph_coef)
        # compute mel spectrogram
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length,
            window=self.windows.get(window_type, self.windows['hamming']))
        spec = np.abs(stft)**2
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            try:
                logspec = librosa.power_to_db(melspec, ref=np.max)
            except:
                logspec = librosa.logamplitude(melspec, ref_power=np.max)
        elif audio_type == 'spectrogram':
            try:
                logspec = librosa.power_to_db(melspec, ref=np.max)
            except:
                logspec = librosa.logamplitude(melspec, ref_power=np.max)
        n_frames = logspec.shape[1]
        if use_raw_length:
            target_length = n_frames
        p = target_length - n_frames
        if p > 0:
            logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
                constant_values=(padval,padval))
        elif p < 0:
            logspec = logspec[:,0:p]
            n_frames = target_length
        logspec = torch.FloatTensor(logspec).permute(1,0)
        return logspec

    def __getitem__(self, index):
        """
        returns: audioA, nframesA, audio, nframesB, textA, textB
        where audioA and audioB are FloatTensors of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform,
              nframesA and nframesB are integers
              textA and textB are strings
        """
        datum = self.data[index]
        wavpath_langA = os.path.join(self.audio_langA_base_path, datum['wav-langA'].replace('.wav', '_one_channel.wav'))
        wavpath_langB = os.path.join(self.audio_langB_base_path, datum['wav-langB'].replace('.wav', '_one_channel.wav'))
        text_langA = datum['text-langA']
        text_langB = datum['text-langB']
        audioA = self._LoadAudio(wavpath_langA)
        audioB = self._LoadAudio(wavpath_langB)
        return audioA, audioB, text_langA, text_langB

    def __len__(self):
        return len(self.data)
