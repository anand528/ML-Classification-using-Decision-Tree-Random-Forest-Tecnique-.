# 🚢 Titanic Survival Prediction – Decision Tree & Random Forest
## 📌 Project Overview
This project applies machine learning classification techniques to the famous Titanic dataset to predict passenger survival. Using Decision Trees and the Random Forest ensemble method, we explore how different features (such as age, gender, class, and fare) influenced survival chances.
The Titanic dataset is a classic benchmark in data science, widely used to demonstrate preprocessing, feature engineering, and model evaluation techniques.

---


## ⚙️ Techniques Used
# Decision Tree Classifier

Simple, interpretable model that splits data based on feature values.
Helps visualize survival rules (e.g., "Females in 1st class had higher survival rates").
Random Forest Classifier
Ensemble of multiple decision trees.
Reduces overfitting and improves accuracy by averaging predictions.
Provides feature importance ranking.
 
---


# 📊 Workflow
### Data Preprocessing

Handling missing values (Age, Cabin, Embarked).
Encoding categorical variables (Sex, Embarked).
Feature scaling and selection.
Exploratory Data Analysis (EDA)
Survival distribution by gender, class, and age.
Correlation heatmaps and feature importance.
Model Training & Evaluation
Train/test split for validation.
Hyperparameter tuning with GridSearchCV.
Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
Results & Insights
Random Forest outperforms a single Decision Tree.
Gender and passenger class are the strongest predictors of survival.

---

## 📈 Key Learnings
# Decision Trees are easy to interpret but prone to overfitting.

Random Forests improve generalization by combining multiple trees.
Feature engineering and preprocessing significantly impact model performance.

---

## 🚀 Future Improvements
# Try other ensemble methods (Gradient Boosting, XGBoost).

Use cross-validation for more robust evaluation.
Deploy the model with a simple web app (Flask/Streamlit).
