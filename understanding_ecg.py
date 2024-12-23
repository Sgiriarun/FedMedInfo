# import pandas as pd

# # Read the first dataset (physiological)
# df_physiological = pd.read_csv('CASE_full/data/interpolated/physiological/sub_25.csv')

# print(df_physiological.head())
# # Get the counts of unique 'video' values from the physiological dataset
# video_counts = df_physiological['video'].value_counts()

# # Read the second dataset (annotations)
# df_annotations = pd.read_csv('CASE_full/data/interpolated/annotations/sub_25.csv')

# # Get the counts of unique 'video' values from the annotations dataset
# annotations_count = df_annotations['video'].value_counts()

# # Create a comparison DataFrame by merging both counts into a single DataFrame
# comparison_df = pd.DataFrame({
#     'Physiological Video Count': video_counts,
#     'Annotations Video Count': annotations_count
# })

# # Fill NaN values with 0 where a video exists in one dataset but not the other
# comparison_df = comparison_df.fillna(0)

# # Print the comparison
# print("Tabular comparison of video counts in both datasets:")
# print(comparison_df)


# import pandas as pd

# # Load the physiological and annotation datasets
# df_physiological = pd.read_csv('extracted_features_sub_25.csv')
# df_annotations = pd.read_csv('CASE_full/data/interpolated/annotations/sub_1.csv')

# # Loop through each unique video ID in both datasets
# for video_id in df_physiological['video'].unique():
#     print(f"\nProcessing video {video_id}...")
    
#     # Filter data for the current video
#     df_physiological_video = df_physiological[df_physiological['video'] == video_id]
#     df_annotations_video = df_annotations[df_annotations['video'] == video_id]
    
#     # Calculate the downsampling factor
#     physiological_samples = len(df_physiological_video)
#     annotation_samples = len(df_annotations_video)
    
#     downsampling_factor = physiological_samples // annotation_samples
#     print(f"Physiological samples: {physiological_samples}, Annotations samples: {annotation_samples}")
#     print(f"Downsampling factor for video {video_id}: {downsampling_factor}")
    
#     # Downsample physiological data
#     df_physiological_resampled = df_physiological_video.iloc[::downsampling_factor]
    
#     # Reset index to align data properly
#     df_physiological_resampled = df_physiological_resampled.reset_index(drop=True)
#     df_annotations_video = df_annotations_video.reset_index(drop=True)
    
#     # Merge the physiological data and annotations for this video
#     df_resampled = pd.concat([df_physiological_resampled, df_annotations_video], axis=1)
    
#     # Print the downsampling factor and a preview of the data
#     print(f"Downsampled physiological data for video {video_id} has {len(df_physiological_resampled)} samples.")
#     print(df_resampled.head())



# import pandas as pd

# # Load the physiological and annotation datasets
# df_physiological = pd.read_csv('extracted_features_sub_25.csv')
# df_annotations = pd.read_csv('CASE_full/data/interpolated/annotations/sub_1.csv')

# # Loop through each unique video ID in both datasets
# for video_id in df_physiological['video'].unique():
#     print(f"\nProcessing video {video_id}...")
    
#     # Filter data for the current video
#     df_physiological_video = df_physiological[df_physiological['video'] == video_id]
#     df_annotations_video = df_annotations[df_annotations['video'] == video_id]
    
#     # Calculate the number of samples
#     physiological_samples = len(df_physiological_video)
#     annotation_samples = len(df_annotations_video)
    
#     print(f"Physiological samples: {physiological_samples}, Annotations samples: {annotation_samples}")
    
#     # Check if annotations exceed physiological samples
#     if annotation_samples > physiological_samples:
#         print(f"Annotations exceed physiological samples for video {video_id}. Upsampling not implemented.")
#         continue  # Skip this video or handle upsampling if needed
    
#     # Calculate the downsampling factor
#     downsampling_factor = max(1, physiological_samples // annotation_samples)  # Ensure the factor is at least 1
#     print(f"Downsampling factor for video {video_id}: {downsampling_factor}")
    
#     # Downsample physiological data
#     df_physiological_resampled = df_physiological_video.iloc[::downsampling_factor]
    
#     # Reset index to align data properly
#     df_physiological_resampled = df_physiological_resampled.reset_index(drop=True)
#     df_annotations_video = df_annotations_video.reset_index(drop=True)
    
#     # Merge the physiological data and annotations for this video
#     merged_samples = min(len(df_physiological_resampled), len(df_annotations_video))
#     df_resampled = pd.concat([df_physiological_resampled.iloc[:merged_samples],
#                               df_annotations_video.iloc[:merged_samples]], axis=1)
    
#     # Print the downsampling factor and a preview of the data
#     print(f"Downsampled physiological data for video {video_id} has {len(df_physiological_resampled)} samples.")
#     print(df_resampled.head())

import numpy as np
import pandas as pd
import os

# Directory paths
npy_directory = '/home/bmi-lab/Documents/fed_emo/processed_features/'
csv_directory = '/home/bmi-lab/Documents/fed_emo/CASE_full/data/interpolated/annotations/'

# Create a list to store the results
results = []

# Loop over all 30 subjects (assuming filenames follow the pattern 'subject_X.npy' and 'sub_X.csv')
for i in range(1, 31):
    # Build the file paths for the .npy and .csv files
    npy_file = os.path.join(npy_directory, f'subject_{i}.npy')
    csv_file = os.path.join(csv_directory, f'sub_{i}.csv')

    # Initialize variables for rows and columns
    npy_rows, npy_columns = None, None
    csv_rows, csv_columns = None, None

    # Load the .npy file and get its shape
    if os.path.exists(npy_file):
        npy_data = np.load(npy_file)
        npy_rows, npy_columns = npy_data.shape
    else:
        npy_rows, npy_columns = 'File not found', 'File not found'

    # Load the .csv file and get its shape
    if os.path.exists(csv_file):
        csv_data = pd.read_csv(csv_file)
        csv_rows, csv_columns = csv_data.shape
    else:
        csv_rows, csv_columns = 'File not found', 'File not found'

    # Append the results for the current subject
    results.append([i, npy_rows, npy_columns, csv_rows, csv_columns])

# Create a DataFrame from the results
df = pd.DataFrame(results, columns=['Subject', 'Numpy Rows', 'Numpy Columns', 'CSV Rows', 'CSV Columns'])

# Print the tabular comparison
print(df)

