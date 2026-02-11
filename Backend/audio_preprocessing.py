import librosa
import numpy as np
import scipy.signal as signal


def gentle_denoise(y, sr):

    sos = signal.butter(4, 80, btype='highpass', fs=sr, output='sos')
    return signal.sosfilt(sos, y)


def windowed_normalization(y, frame_length=2048):

    rms = librosa.feature.rms(y=y, frame_length=frame_length)[0]
    rms[rms == 0] = 1e-6
    gain = np.repeat(rms, frame_length)[:len(y)]
    return y / gain * np.mean(rms)


def compute_jitter_shimmer(y, sr):

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=75,
        fmax=300,
        sr=sr
    )

    f0 = f0[~np.isnan(f0)]
    jitter = np.mean(np.abs(np.diff(f0))) if len(f0) > 1 else 0.0

    frame_amplitude = librosa.feature.rms(y=y)[0]
    shimmer = np.mean(np.abs(np.diff(frame_amplitude))) if len(frame_amplitude) > 1 else 0.0

    return jitter, shimmer


def preprocess_audio(
    file_path,
    sr=16000,
    vad_top_db=30,
    min_pause_sec=0.15
):

    y, sr = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)

    y = gentle_denoise(y, sr)

    intervals = librosa.effects.split(y, top_db=vad_top_db)

    total_duration = len(y) / sr
    speech_duration = np.sum([(end - start) / sr for start, end in intervals])
    phonation_rate = speech_duration / total_duration

    pauses = []
    intra_sentence_pauses = 0
    prev_end = 0

    for start, end in intervals:
        pause = (start - prev_end) / sr
        if pause > min_pause_sec:
            pauses.append(pause)
            if pause < 1.0:
                intra_sentence_pauses += 1
        prev_end = end

    mean_pause = np.mean(pauses) if pauses else 0.0
    pause_std = np.std(pauses) if pauses else 0.0
    pause_count = len(pauses)

    y_speech = np.concatenate([y[start:end] for start, end in intervals]) \
               if intervals.any() else y

    y_speech = windowed_normalization(y_speech)

    jitter, shimmer = compute_jitter_shimmer(y_speech, sr)

    prosody_vector = np.array([
        mean_pause,            
        pause_std,              
        pause_count,            
        intra_sentence_pauses,  
        phonation_rate,         
        jitter,                 
        shimmer                 
    ], dtype=np.float32)

    return y_speech, sr, prosody_vector
