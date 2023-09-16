# Use this file to augment the original data
# The original data only includes 100 entries per label
# You may add noise by update the noise_scale value
# You may also add data through data augmentation by revising the number on the augmented_subset

import pandas as pd
import numpy as np

# Read the csv file
df = pd.read_csv('crop_recommendation_orig.csv')

# Define the function to augment data
def augment_data(subset, n_times, noise_scale=0.05): # 5% noise scale
    augmented_list = [subset]

    for _ in range(n_times-1):
        noise = np.random.normal(loc=0, scale=noise_scale, size=subset.shape)
        new_data = subset.copy()

        # Alter numerical columns
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            if col in ['N', 'P', 'K']:
                new_data[col] = np.round(new_data[col] + noise[:, list(subset.columns).index(col)])
                # Ensure values are not negative
                new_data[col] = new_data[col].clip(0)
            else:
                new_data[col] += noise[:, list(subset.columns).index(col)]
        
        augmented_list.append(new_data)

    return pd.concat(augmented_list)

# Unique labels in the dataset
labels = df['label'].unique()

# List to hold augmented data subsets
augmented_data_list = []

# Loop through labels and apply augmentation
for label in labels:
    subset = df[df['label'] == label]
    augmented_subset = augment_data(subset, 50)
    augmented_data_list.append(augmented_subset)

# Concatenate all augmented subsets
augmented_df = pd.concat(augmented_data_list)

# Save augmented dataframe to a new csv
augmented_df.to_csv('augmented_file.csv', index=False)

