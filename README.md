# Heart Disease Prediction Using Logistic Regression

## Overview
This project focuses on building a machine learning model to predict the presence of heart disease using Logistic Regression. It uses a dataset containing various medical attributes of patients and aims to assist in early diagnosis based on clinical data.

## Objectives
- Predict the likelihood of heart disease in a patient
- Use logistic regression as a baseline classification model
- Analyze important health features contributing to the prediction

## Dataset
- The dataset typically includes:
  - Age
  - Sex
  - Chest pain type
  - Resting blood pressure
  - Cholesterol
  - Fasting blood sugar
  - Resting ECG results
  - Maximum heart rate achieved
  - Exercise-induced angina
  - ST depression induced by exercise
  - Slope of peak exercise ST segment
  - Number of major vessels colored by fluoroscopy
  - Thalassemia
  - Target (0 = no disease, 1 = disease)

## Tools & Libraries
- Python
- Pandas, NumPy – data handling
- Matplotlib, Seaborn – data visualization
- Scikit-learn – modeling and evaluation

## Workflow
1. **Data Loading & Exploration**
   - Check for missing values and understand feature distributions.
2. **Data Preprocessing**
   - Encode categorical variables.
   - Feature scaling (if needed).
3. **Modeling**
   - Train a Logistic Regression model.
   - Evaluate performance using accuracy, precision, recall, and F1-score.
4. **Visualization**
   - Plot confusion matrix, ROC curve, feature correlations.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook
   ```

4. Open `Heart Disease Prediction Using Logistic Regression.ipynb` and follow along.

## Results
- Logistic Regression achieved reasonable performance for baseline classification.
- Key indicators included chest pain type, cholesterol levels, and maximum heart rate.

## Future Improvements
- Try other models: Random Forest, XGBoost, or Neural Networks.
- Perform hyperparameter tuning with GridSearchCV.
- Deploy the model using Flask or Streamlit for interactive predictions.

## License
This project is licensed under the MIT License.
