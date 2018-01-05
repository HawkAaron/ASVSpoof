# -*- coding: utf-8 -*-
import librosa
import numpy as np
import pywt

sample_rate = 16000
n_fft = int(25 * sample_rate / 1000)
hop_length = int(5 * sample_rate / 1000)

n_cqt = 96
f_max = sample_rate / 2
f_min = f_max / (2 ** 9)

def feat_window(feat):
    window = 6  # actual size = 2 * window - 1
    feat = np.pad(feat, [[0, 0], [window - 1, window - 1]], mode='edge')
    tmp_feat = []
    for i in range(window - 1, feat.shape[1] - window, 2*window-1):
        fram_win = feat[:, i-window+1: i+window].reshape(-1) # store feat_dim * 11 map as vector
        tmp_feat.append(fram_win)
    return np.array(tmp_feat, dtype=np.float32)

def feat_padding(feat, width=400):
    dim = feat.shape[0]

    padding = width - feat.shape[1] % width
    if padding < 200:
        feat = np.pad(feat, [[0, 0], [0, padding]], mode='edge')
    if feat.shape[1] % width:
        feat = feat[:, :-(feat.shape[1]%width)]
    tmp_feat = feat.T.reshape(-1, width, dim)

    return tmp_feat

def extract(wav_path, feat_type='db4'):
    if feat_type == 'db4':
        return extract_db4(wav_path)
    if feat_type == 'db8':
        return extract_db8(wav_path)
    if feat_type == 'cqcc':
        return extract_cqcc(wav_path)
    if feat_type == 'fft':
        return extract_fft(wav_path)
    raise NotImplementedError('No such feature type')

def extract_db4(wav_path):
    y, sr = librosa.load(wav_path, sr=sample_rate)
    S, _ = librosa.core.spectrum._spectrogram(y, hop_length=50, n_fft=799, power=2)
    # S = librosa.power_to_db(S)
    cA, cD = pywt.dwt(S, 'db4')
    return feat_window(cA)

def extract_db8(wav_path):
    raise NotImplementedError('Not implement')

def extract_cqcc(wav_path):
    y, sr = librosa.load(wav_path, sr=sample_rate)
    cqt = librosa.feature.chroma_cqt(y, sr, hop_length=hop_length, fmin=f_min, n_chroma=n_cqt, n_octaves=5)
    return cqt.T

def extract_fft(wav_path, hop_length=20, n_fft=798):
    y, _ = librosa.load(wav_path, sr=16000)
    S, _ = librosa.core.spectrum._spectrogram(y, hop_length=hop_length, n_fft=n_fft, power=2)
    return feat_padding(S)