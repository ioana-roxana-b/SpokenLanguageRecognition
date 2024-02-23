import os
import numpy as np
import pandas as pd
import features

def create_dataset(folder_path, label):
    data = []
    feature_names = None
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            audio_path = os.path.join(folder_path, filename)
            audio_features = features.extract_features(audio_path)

            if any(value != 0 for value in audio_features.values()):
                audio_features_array = np.array(list(audio_features.values()))
                row_data = np.concatenate([audio_features_array, [label]])
                data.append(row_data)

                if feature_names is None:
                    feature_names = list(audio_features.keys())

    return data, feature_names

def create_and_append_dataset(folder_paths, output_csv="all_features.csv"):
    all_data = []
    all_feature_names = []

    for label, folder_path in enumerate(folder_paths):
        data, feature_names = create_dataset(folder_path, label)
        all_data += data
        all_feature_names = feature_names

    if not all_data:
        print("No data found. Exiting.")
        return

    try:
        df = pd.DataFrame(all_data, columns=all_feature_names + ["label"])
        df.to_csv(output_csv, index=False)
        print(f"Dataset appended to {output_csv}")
    except Exception as e:
        print(f"Error creating the DataFrame: {e}")

