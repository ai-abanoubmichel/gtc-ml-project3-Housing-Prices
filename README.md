# 🏠 gtc_ml_project3_Housing_Prices

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Processing-lightblue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-darkblue?logo=plotly)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal?logo=seaborn)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoosting-orange?logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-GradientBoosting-green?logo=lightgbm)
![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-yellow?logo=catboost)
![EDA](https://img.shields.io/badge/EDA-Exploratory%20Data%20Analysis-brightgreen)
![Machine Learning](https://img.shields.io/badge/ML-Regression-blue?logo=mlflow)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview
This is my **third project** during the **Machine Learning Internship at Genius Technology Center (GTC)**.  
The **main goal** of this project was to predict the **median house values in California** using **machine learning regression models**.  

The process began with **data cleaning and EDA**, followed by training multiple models (from linear regression to advanced boosting methods), and finally tuning hyperparameters for the best performance.  

---

## 📊 Dataset Overview
- **Target Variable:** `median_house_value`  
- **Features:** `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `latitude`, `longitude`, and others.  
- **Size:** ~20,000 records  

---

## 🛠️ Methodology & Workflow
1. **Data Cleaning**
   - Handled missing values
   - Removed inconsistencies
2. **Exploratory Data Analysis (EDA)**
   - Distribution of housing values
   - Correlation between features
   - Impact of median income & location on housing price
3. **Feature Scaling & Splitting**
   - Normalization/standardization
   - Train-test split (80%-20%)
4. **Model Training**
   - **Linear Models:** Linear Regression
   - **Tree-based Models:** Decision Tree, Random Forest
   - **Distance-based:** KNN
   - **Boosting Models:** AdaBoost, CatBoost, XGBoost, LightGBM
   - **Support Vector Regressor (SVR)**
5. **Evaluation**
   - Metrics: **R² Score** & **Mean Absolute Error (MAE)**
   - Comparison of train vs. test performance
6. **Hyperparameter Tuning**
   - GridSearchCV for best parameters
   - Best results achieved with **CatBoost**

---

## 🚀 Results 
- **Best Model:** CatBoost Regressor  
- **Best Parameters:** `depth=8`, `iterations=500`, `learning_rate=0.1`  
- **Performance:**  
  - High R² score on test data  
  - Low mean absolute error compared to other models  

📊 Example of comparison table:

| Model              | Train R² | Test R² | MAE   |
|--------------------|----------|---------|-------|
| Linear Regression  | 0.62     | 0.58    | 48k   |
| Decision Tree      | 0.95     | 0.62    | 42k   |
| Random Forest      | 0.92     | 0.75    | 35k   |
| **CatBoost** ✅     | **0.94** | **0.82**| **28k** |

---

## 📊 Visualizations (Sample)

### 1. Distribution of House Values
![House Values Distribution](Images/HouseValue_Distribution.png)

### 2. Correlation Heatmap
![Correlation Heatmap](Images/Correlation_Heatmap.png)

### 3. Median Income vs Median House Value
<p align="center">
  <img src="Images/Income_vs_Price1.png" alt="Income vs Price 1" width="45%"/>
  <img src="Images/Income_vs_Price2.png" alt="Income vs Price 2" width="45%"/>
</p>

---

## 💡 Key Learnings
- Boosting models (CatBoost/XGBoost/LightGBM) significantly outperform simpler models.  
- Proper feature scaling and handling missing values are crucial for regression tasks.  
- Median income proved to be the strongest predictor of house prices.  
- Hyperparameter tuning can drastically improve results compared to default settings.  

---

## 📂 Repository Structure
