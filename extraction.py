# import pandas as pd
# import numpy as np

# # Read the physiological dataset
# df_physiological = pd.read_csv('CASE_full/data/interpolated/physiological/sub_25.csv')

# # Define the window size (50 samples)
# window_size = 50

# # Initialize an empty list to store the extracted features
# features = []

# # Loop through the dataset in steps of `window_size`
# for i in range(0, len(df_physiological), window_size):
#     # Extract the current window
#     window = df_physiological.iloc[i:i + window_size]

#     # Skip the last window if it's smaller than the desired size
#     if len(window) < window_size:
#         break

#     # Calculate features for the current window
#     feature = {
#         'mean_HR': window['ecg'].mean(),  # ECG mean HR
#         'SDNN': window['bvp'].std(),      # BVP SDNN
#         'mean_SCR': window['gsr'].mean(), # GSR mean SCR
#         'mean_RR': window['rsp'].mean(),  # Respiration mean RR
#         'SDT': window['skt'].std(),       # Skin Temperature SDT
#         # 'mean_Zygo': window['emg_zygo'].mean(), # EMG-zygomaticus mean amplitude
#         # 'mean_Corr': window['emg_coru'].mean(), # EMG-corrugator mean amplitude
#         # 'mean_Trap': window['emg_trap'].mean(), # EMG-trapezius mean amplitude
#         # 'video': window['video'].iloc[0]  # Video ID (assume constant within a window)
#     }

#     # Append the feature dictionary to the list
#     features.append(feature)

# # Create a DataFrame from the features list
# features_df = pd.DataFrame(features)

# # Print some of the extracted features
# print(features_df.head())

# # Save the features to a CSV file
# # features_df.to_csv('extracted_features_sub_25.csv', index=False)

import pandas as pd
import numpy as np
import os

# Directories
physiological_directory = 'CASE_full/data/interpolated/physiological/'
annotation_directory = 'CASE_full/data/interpolated/annotations/'
output_directory = 'processed_features/'

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Define window size
window_size = 50

# Feature extraction function
def extract_features(window):
    """Extract features from a 50-sample window."""
    return {
        'mean_HR': window['ecg'].mean(),            # ECG mean HR
        'SDNN': window['bvp'].std(),               # BVP SDNN
        'mean_SCR': window['gsr'].mean(),          # GSR mean SCR
        'mean_RR': window['rsp'].mean(),           # Respiration mean RR
        'SDT': window['skt'].std(),                # Skin Temperature SDT
    }

# Loop through each subject
for subject_id in range(1, 31):  # Assuming 30 subjects
    print(f"Processing Subject {subject_id}...")

    # Load physiological and annotation data
    physiological_file = os.path.join(physiological_directory, f'sub_{subject_id}.csv')
    annotation_file = os.path.join(annotation_directory, f'sub_{subject_id}.csv')

    df_physiological = pd.read_csv(physiological_file)
    df_annotations = pd.read_csv(annotation_file)

    # Initialize list to store features and labels
    subject_features = []

    # Loop through physiological data in 50-sample windows
    for i in range(0, len(df_physiological), window_size):
        # Extract the current window
        window = df_physiological.iloc[i:i + window_size]

        # Skip incomplete windows
        if len(window) < window_size:
            break

        # Extract features for the window
        features = extract_features(window)

        # Align with annotations (assuming the first annotation corresponds to the window)
        annotation_idx = i // window_size
        if annotation_idx < len(df_annotations):
            labels = df_annotations.iloc[annotation_idx][['valence', 'arousal']].to_dict()
        else:
            print(f"Warning: Missing annotation for window starting at {i} in Subject {subject_id}.")
            continue

        # Combine features and labels
        features.update(labels)
        print(features)
        # Add to the subject's feature list
        subject_features.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(subject_features)

    # Save as .npy file
    output_file = os.path.join(output_directory, f'subject_{subject_id}.npy')
    np.save(output_file, features_df.to_numpy())

    print(f"Saved processed features for Subject {subject_id} to {output_file}")
