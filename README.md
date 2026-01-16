# 🚀 Credit Card Fraud Detection: Optimised XGBoost Pipeline

## 📋 Project Overview
This project develops a robust machine learning pipeline to identify fraudulent credit card transactions. Operating within the financial sector, the primary challenge addressed is the extreme **class imbalance**—where fraudulent activities represent less than 0.2% of the total dataset. 

The model is specifically **prioritised for Recall (Catch Rate)**. In fraud detection, the goal is to be highly sensitive to the positive class (Class 1) to minimise the risk of missing critical cases (False Negatives), as a missed fraud is significantly more costly than a false alarm.

## 📊 Final Performance Results
The model was evaluated using a held-out test set to ensure stability and real-world applicability.

| Metric | Score |
| :--- | :--- |
| **Final Recall (Class 1)** | **86.73%** |
| **Overall Accuracy** | 98.00% |
| **Precision (Class 1)** | 0.09 |
| **Features Retained** | 29 |

### 🔍 Key Insights
* **Catch Rate**: The model successfully identifies **~87%** of all actual fraudulent cases.
* **Precision Trade-off**: The low precision (0.09) is a deliberate consequence of the high-recall priority. We accept more "False Alarms" to ensure actual fraud does not go undetected.
* **Dimensionality**: By reducing the feature set to the 29 most influential variables, the model remains lean and maintainable without losing predictive power.



---

## 🛠️ Methodology
1.  **Exploratory Data Analysis (EDA)**: Conducted an initial audit to understand distribution and class imbalance.
2.  **Class Imbalance Mitigation**: Utilised **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training set.
3.  **Model Selection**: An **XGBoost Classifier** was chosen after benchmarking against Logistic Regression and Random Forest.
4.  **Feature Engineering**: 
    * **Correlation Analysis**: Removed redundant features with a correlation threshold $> 0.85$.
    * **Feature Importance**: Used "Gain" scores to identify the top 29 predictors.
5.  **Validation**: Performed manual Cross-Validation to ensure consistent performance across data splits.

---

## 📂 Data Source
The dataset used is the **Credit Card Fraud Detection** dataset from Kaggle.
* **Link:** [Kaggle Dataset - Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Note:** Features V1-V28 are PCA-transformed for privacy. 'Time' and 'Amount' are the only non-transformed variables.

---

## 💻 How to Use the Model
To run this model, ensure you have `xgboost`, `joblib`, and `pandas` installed.

### Loading the Model and Features
```python
import joblib
import pandas as pd

# 1. Load the finalised model and the required feature list
model = joblib.load('models/optimized_recall_model.pkl')
model_features = joblib.load('models/model_features.joblib')

# 2. Filter your new data for the 29 optimised features
# new_data_filtered = new_data[model_features]

# 3. Predict
# predictions = model.predict(new_data_filtered)
