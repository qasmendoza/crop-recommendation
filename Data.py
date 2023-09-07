import pandas as pd
import numpy as np

# Read the csv file
df = pd.read_csv('crop_recommendation_orig.csv')

# Define the function to augment data
def augment_data(subset, n_times):
    augmented = subset.copy()
    for _ in range(n_times-1):
        noise = np.random.normal(loc=0, scale=0.10, size=subset.shape)
        new_data = subset.copy()

        # Alter numerical columns
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            if col in ['N', 'P', 'K']:
                # Round the values for these columns after adding noise
                new_data[col] = np.round(new_data[col] + noise[:, list(subset.columns).index(col)])
            else:
                new_data[col] += noise[:, list(subset.columns).index(col)]
        
        augmented = pd.concat([augmented, new_data])
    return augmented

# Unique labels in the dataset
labels = df['label'].unique()

# Create an empty dataframe to hold augmented data
augmented_df = pd.DataFrame()

# Loop through labels and apply augmentation
for label in labels:
    subset = df[df['label'] == label]
    augmented_subset = augment_data(subset, 20) # x20 the data
    augmented_df = pd.concat([augmented_df, augmented_subset])

# Save augmented dataframe to a new csv
augmented_df.to_csv('augmented_file.csv', index=False)

