# Crop Recommendation Project 🌾🌽🥔🍇🍌🍎 

A POC (Proof of Concept) project that utilizes Machine Learning and Data Science to advise on the ideal crops to plant, the right fertilizers to employ, and to detect diseases that might afflict your crops.

# ⚠️🛑 Disclaimer! 🛑⚠️

**This project is a proof of concept only**. The data offered in this context comes without assurances from its author. It is not advised to base farming decisions on it. The author will not be held accountable for consequences arising from its use. Still, this project showcases how Machine Learning and Data Science can effectively apply to precision farming when grounded in comprehensive and validated data.

# Procedure 📖📖

- The original dataset is from [Kaggle](https://www.kaggle.com/code/atharvaingle/what-crop-to-grow/input). This dataset only consists of 100 entries per crop.
- I created a Python file entitled [data.py](https://github.com/qasmendoza/crop-recommendation/blob/main/Data.py) to augment the orignal dataset. The augmentation steps include introducing 10% noise data and mutiplying the dataset by 50x per crop.
- The file [Crops.ipynb](https://github.com/qasmendoza/crop-recommendation/blob/main/Crops.ipynb) was created for the following steps:
  -  I used the augmented dataset for data analysis and visualization.
  -  Feature Engineering
  -  Data Splitting
  -  Modeling
  -  Model Evaluation and Validation
- For Machine Learning models, the following classifiers were used:
  - Random Forest
  - KNN
  - Naive Bayes
  - Logistic Regression
  - XGBoost

# Metrics 📈📊📉

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Random Forest        | 0.9613   | 0.9700    | 0.9613 | 0.9566   |
| KNN                  | 0.9570   | 0.9621    | 0.9570 | 0.9568   |
| Naive Bayes          | 0.9570   | 0.9770    | 0.9760 | 0.9760   |
| Logistic Regression  | 0.9069   | 0.9119    | 0.9069 | 0.9060   |
| XGBoost              | 0.9740   | 0.9759    | 0.9740 | 0.9739   |

# Making Predictions 🔮🧙🏽✨
- The code will ask for entries: N, P, K, Temperature (C), Humidity, Ph, and Rainfall.
- The model will use ensemble prediction and voting.
  

<br>
ENSEMBLE PREDICTION RESULT      

| Metric           | Value           |
|------------------|-----------------|
| Crop Predicted   | Banana          |
| Accuracy         | 0.9488          |
| Precision        | 0.9709          |
| Recall           | 0.9488          |
| F1 Score         | 0.9425          |
--------------------------------------





