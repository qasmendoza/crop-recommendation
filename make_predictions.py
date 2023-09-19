import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def ensemble_predict(classifiers, data_df, le=None):
    """Function to get ensemble prediction."""
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

def calculate_ensemble_metrics(classifiers, X_test, y_test, le=None):
    """Function to calculate ensemble metrics."""
    y_preds = [ensemble_predict(classifiers, pd.Series(row), le) for row in X_test.values]
    metrics = {
        'Accuracy': accuracy_score(y_test, y_preds),
        'Precision': precision_score(y_test, y_preds, average='weighted'),
        'Recall': recall_score(y_test, y_preds, average='weighted'),
        'F1 Score': f1_score(y_test, y_preds, average='weighted')
    }
    return metrics

def predict_crop_value(N, P, K, temperature, humidity, ph, rainfall, classifiers, le, X_test, y_test):
    """Function to predict crop value and collect metrics."""
    try:
        original_feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        data = [[N, P, K, temperature, humidity, ph, rainfall]]
        data_df = pd.DataFrame(data, columns=original_feature_names)

        # Using the ensemble prediction function
        ensemble_prediction = ensemble_predict(classifiers, data_df, le)

        # Getting metrics using the function
        ensemble_metrics = calculate_ensemble_metrics(classifiers, X_test, y_test, le)

        # Collecting model-wise predictions for display
        model_predictions = {}
        for clf_name, clf in classifiers.items():
            proba = clf.predict_proba(data_df.values)
            predicted_index = np.argmax(proba)
            if clf_name == "XGBoost" and le:
                predicted_label = le.inverse_transform([predicted_index])[0]
            else:
                predicted_label = clf.classes_[predicted_index]
            model_predictions[clf_name] = predicted_label

        return ensemble_prediction, ensemble_metrics, model_predictions

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None
