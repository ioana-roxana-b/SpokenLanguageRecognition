import pandas as pd
import librosa
import numpy as np

def extract_features(audio_path, max_length=500000):
    audio, _ = librosa.load(audio_path, sr=22050)

    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=10).flatten()
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=22050)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=22050)[0]
    formants = np.sum(librosa.effects.harmonic(audio), axis=0)
    pitch, _ = librosa.core.piptrack(y=audio)
    pitch = pitch[pitch > 0]
    pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0  # Handle the case when pitch is empty
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=22050)

    # Pad or truncate features to a fixed length
    mfccs = pad_or_truncate(mfccs, max_length)
    spectral_centroid = pad_or_truncate(spectral_centroid, max_length)
    spectral_bandwidth = pad_or_truncate(spectral_bandwidth, max_length)
    formants = pad_or_truncate(formants, max_length)
    pitch_mean = np.array([pitch_mean])
    mel_spectrogram = pad_or_truncate(mel_spectrogram.flatten(), max_length)

    mean = np.mean(audio)
    median = np.median(audio)
    std_dev = np.std(audio)

    # Convert scalar features to 1D arrays
    mean = np.array([mean])
    median = np.array([median])
    std_dev = np.array([std_dev])

    feature_names = ( ["mfcc_" + str(i) for i in range(len(mfccs))] +
                      ["spectral_centroid", "spectral_bandwidth", "formants", "pitch_mean"]
                     + ["mean", "median", "std_dev"] + [f"mel_spectrogram_{i}" for i in range(len(mel_spectrogram.flatten()))])

    feature_dict = dict(zip(feature_names, np.concatenate([
        mfccs,
        spectral_centroid,
        spectral_bandwidth,
        formants,
        pitch_mean,
        mean,
        median,
        std_dev,
        mel_spectrogram.flatten()
    ])))

    return feature_dict

def pad_or_truncate(feature, max_length):
    if np.isscalar(feature):
        return np.array([feature] * max_length)
    elif len(feature) < max_length:
        return np.pad(feature, (0, max_length - len(feature)))
    elif len(feature) > max_length:
        return feature[:max_length]
    else:
        return feature


def create_and_append_dataset(folder_paths, output_csv):
    all_data = []

    for folder_path in folder_paths:
        data = extract_features(folder_path)
        all_data.append(data)

    if not all_data:
        print("No data found. Exiting.")
        return

    feature_names = list(all_data[0].keys())

    try:
        df = pd.DataFrame(all_data, columns=feature_names)
        df.to_csv(output_csv, index=False)
        print(f"Dataset appended to {output_csv}")
    except Exception as e:
        print(f"Error creating the DataFrame: {e}")


