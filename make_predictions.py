# make_predictions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def ensemble_predict(classifiers, data_df, le=None):
    if len(data_df.shape) == 1:
        data_df = data_df.values.reshape(1, -1)
    else:
        data_df = data_df.values

    votes = []
    for clf_name, clf in classifiers.items():
        proba = clf.predict_proba(data_df)
        predicted_index = np.argmax(proba)

        if clf_name == "XGBoost" and le:
            predicted_label = le.inverse_transform([predicted_index])[0]
        else:
            predicted_label = clf.classes_[predicted_index]
        votes.append(predicted_label)

    vote_counts = Counter(votes)
    most_common_votes = vote_counts.most_common()
    if len(most_common_votes) > 1 and most_common_votes[0][1] == most_common_votes[1][1]:
        tied_classes = [vote[0] for vote in most_common_votes if vote[1] == most_common_votes[0][1]]
        return tied_classes[0], votes
    else:
        return most_common_votes[0][0], votes

def predict_crop_value(N, P, K, temperature, humidity, ph, rainfall, classifiers, le):
    original_feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    data = [[N, P, K, temperature, humidity, ph, rainfall]]
    data_df = pd.DataFrame(data, columns=original_feature_names)

    ensemble_prediction, votes = ensemble_predict(classifiers, data_df, le)
    
    # Visualization and Outputs
    print("Model-wise Predictions:")
    print("| Model Name            | Prediction   |")
    print("|-----------------------|--------------|")

    for clf_name, clf in classifiers.items():
        proba = clf.predict_proba(data_df.values)
        predicted_index = np.argmax(proba)

        if clf_name == "XGBoost" and le:
            predicted_label = le.inverse_transform([predicted_index])[0]
        else:
            predicted_label = clf.classes_[predicted_index]
        print(f"| {clf_name:<21} | {predicted_label:<12} |")

    vote_counts = Counter(votes)
    plt.figure(figsize=(10,6))
    colors = ['blue', 'green', 'red', 'cyan', 'purple']
    plt.bar(vote_counts.keys(), vote_counts.values(), color=colors, alpha=0.75)
    plt.yticks(np.arange(0, 6, 1))
    plt.ylim(0, 5)
    plt.ylabel('Number of Models Voting', fontsize=12)
    plt.xlabel('Crops', fontsize=12)
    plt.title('Number of Model Votes for Each Predicted Crop', fontsize=14)
    plt.xticks(rotation=0, fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.margins(0.1)
    plt.tight_layout()
    plt.show()

    print("\n--------------------------------------------------")
    print("              ENSEMBLE PREDICTION RESULT      ")
    print("--------------------------------------------------")
    print(f"| Crop Predicted   | {ensemble_prediction:<27} |")
    print("|------------------|-----------------------------|")
    print("--------------------------------------------------")
    print("\nNote:")
    print("The ensemble prediction is derived from a majority consensus across various models.")

    return ensemble_prediction
