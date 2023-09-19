import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensemble Prediction Function
def ensemble_predict(classifiers, data_df, le=None):
    # Convert data_df to 2D if it's 1D
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
        return tied_classes[0]
    else:
        return most_common_votes[0][0]

# Ensemble Metrics Calculation
def calculate_ensemble_metrics(classifiers, X_test, y_test, le=None):
    y_preds = [ensemble_predict(classifiers, pd.Series(row), le) for row in X_test.values]
    metrics = {
        'Accuracy': accuracy_score(y_test, y_preds),
        'Precision': precision_score(y_test, y_preds, average='weighted'),
        'Recall': recall_score(y_test, y_preds, average='weighted'),
        'F1 Score': f1_score(y_test, y_preds, average='weighted')
    }
    return metrics

def predict_crop_value(N, P, K, temperature, humidity, ph, rainfall, classifiers, le, X_test, y_test):
    try:
        original_feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        data = [[N, P, K, temperature, humidity, ph, rainfall]]
        data_df = pd.DataFrame(data, columns=original_feature_names)

        # Using the ensemble prediction function
        ensemble_prediction = ensemble_predict(classifiers, data_df, le)

        # Getting metrics using the function
        ensemble_metrics = calculate_ensemble_metrics(classifiers, X_test, y_test, le)

        # Tabular Representation of Model-wise Predictions
        print("Model-wise Predictions:")
        print("| Model Name            | Prediction   |")
        print("|-----------------------|--------------|")

        votes = []  # List to collect individual model predictions

        for clf_name, clf in classifiers.items():
            proba = clf.predict_proba(data_df.values)
            predicted_index = np.argmax(proba)
    
            if clf_name == "XGBoost" and le:
                predicted_label = le.inverse_transform([predicted_index])[0]
            else:
                predicted_label = clf.classes_[predicted_index]
            print(f"| {clf_name:<21} | {predicted_label:<12} |")
    
            votes.append(predicted_label)  # Add the model's prediction to the votes list

        # Visualization: Histogram of Predictions
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

        # Tabular Representation of Ensemble Prediction Results
        print("\n--------------------------------------------------")
        print("              ENSEMBLE PREDICTION RESULT      ")
        print("--------------------------------------------------")
        print(f"| Crop Predicted   | {ensemble_prediction:<27} |")
        print("|------------------|-----------------------------|")
        print(f"| Accuracy         | {ensemble_metrics['Accuracy']:<27} |")
        print(f"| Precision        | {ensemble_metrics['Precision']:<27} |")
        print(f"| Recall           | {ensemble_metrics['Recall']:<27} |")
        print(f"| F1 Score         | {ensemble_metrics['F1 Score']:<27} |")
        print("--------------------------------------------------")
        print("\nNote:")
        print("The ensemble prediction is derived from a majority consensus across various models.")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return ensemble_prediction
