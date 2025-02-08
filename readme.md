# 🏡 Sport injury prediction
Predicting injury beforehand would be a huge help to the players, ultimately revolutionising the sports industry.Knowing the resting period in advance, would even help teams strategize in a better manner for future tournaments.We also aim to tell the players in advance about the body part more likely to be injured so that the players can take prior measures in order to prevent those injuries.

This project predicts in Sport injury pridection using a Random Forest Regression model and k nearest nighboring. The workflow includes data preprocessing, visualization, model training, hyperparameter tuning, and evaluation.

## 📌 Project Overview

The dataset (sports_injury1.csv) contains 1100 rows and 7 columns, with features such as:

- Numerical: Age,Training_Hours_per_Week,Previous_Injuries,BMI.
- Categorical:Gender,Sport,Injury_Risk

The goal is to train a robust model that accurately predicts Injury_Risk based on these features.

## 🛠 Project Workflow

1. Data Loading & Exploration – Import the dataset, check structure, and identify key features.
2. Preprocessing & Feature Engineering – Handle missing values, outliers, and encode categorical variables.
3. Data Visualization & Analysis – Use Pair Plot, box plots, and correlation heatmaps to explore relationships.
4. Model Implementation & Training – Split data, preprocess features, and train a Random Forest Regressor and knn.
5. Hyperparameter Tuning – Optimize the model using Grid Search for the best performance.
6. Model Evaluation – Assess performance using MAE, MSE, RMSE, and R² metrics.

## 📂 Key Files

- 📜 `mach1.ipynb` – Jupyter Notebook containing the full implementation.
- 📊 `des.csv` – The dataset used for training and testing.

## 🚀 How to Run

1. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn fastapi uvicorn joblib pydantic python-multipart

   ```

2. Open the Jupyter Notebook (mach1.ipynb) in Jupyter Lab or Notebook.

3. Run each cell sequentially to execute the code.

4. View results, including visualizations and model evaluation metrics.
