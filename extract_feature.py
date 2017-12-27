# -*- coding: utf-8 -*-
import librosa
import numpy as np
import pywt

sr = 16000
n_fft = int(25 * sr / 1000)
hop_length = int(5 * sr / 1000)

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
    y, sr = librosa.load(wav_path, sr=16000)
    S, _ = librosa.core.spectrum._spectrogram(y, hop_length=50, n_fft=799, power=2)
    # S = librosa.power_to_db(S)
    cA, cD = pywt.dwt(S, 'db4')
    return cA

def extract_db8(wav_path):
    raise NotImplementedError('Not implement')

def extract_cqcc(wav_path):
    raise NotImplementedError('Not implement')

def extract_fft(wav_path, hop_length=20, n_fft=798):
    y, _ = librosa.load(wav_path, sr=16000)
    S, _ = librosa.core.spectrum._spectrogram(y, hop_length=hop_length, n_fft=n_fft, power=2)
    return S