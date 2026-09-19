# Car Price Prediction

A machine learning regression project for predicting used car prices based on vehicle characteristics such as production year, mileage, engine volume, fuel type, gearbox, drive wheels, and other features.

The project covers an end-to-end machine learning workflow, including data preprocessing, feature engineering, categorical encoding, feature scaling, model comparison using cross-validation, hyperparameter tuning, and final model evaluation.

## Project Overview

The goal of this project is to build a regression model capable of predicting the price of a used car based on its available features.

## Dataset

The dataset used in this project is the **Car Price Prediction Challenge** dataset from Kaggle. It contains **19,237 records and 18 columns** before preprocessing.

**Source:** [Car Price Prediction Challenge](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge/data)

## Data Preprocessing

The project includes several data preprocessing and feature engineering steps:

* Removing duplicate records and the `ID` column
* Handling missing values
* Cleaning and converting numerical features
* Handling categorical features
* Handling outliers
* Combining car manufacturer and model information
* Target Encoding for the high-cardinality `Car` feature (applied to the full dataset before the train/test split)
* Inspecting unusually low prices (1,633 records, about 8.6% of the data, priced below 500). These cars have the same median production year as the rest of the data (2012) and only moderately different mileage (median of about 149,000 km compared with 125,000 km) and engine volume (2.4 compared with 2.0), so no clear pattern explained the low prices and the records were kept
* Removing an extreme price outlier

After preprocessing and encoding, the dataset contained **18,923 records and 52 columns** (51 features plus the target).

## Model Training

The data was split into training and testing sets using a **75/25 split**.

Nine regression models were tuned using `GridSearchCV` with **5-Fold Cross-Validation** and **R²** as the evaluation metric:

* Linear Regression
* SGD Regressor
* Lasso Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Decision Tree Regressor
* K-Nearest Neighbors Regressor
* XGBoost Regressor

The features were standardized using `StandardScaler` (fitted on the training set only) for the scale-sensitive models: SGD Regressor, Lasso, Ridge, and K-Nearest Neighbors.

### Cross-Validation Results

| **Model**                       | **Best CV R²** |
| ------------------------------- | -------------: |
| Linear Regression               |         0.2623 |
| SGD Regressor                   |         0.2623 |
| Lasso Regression                |         0.2623 |
| Ridge Regression                |         0.2624 |
| Random Forest Regressor         |         0.7175 |
| **Gradient Boosting Regressor** |     **0.7581** |
| Decision Tree Regressor         |         0.6434 |
| K-Nearest Neighbors Regressor   |         0.5500 |
| XGBoost Regressor               |         0.7478 |

The linear models performed far below the tree-based models, which suggests that the relationship between the features and the price is not linear.

## Final Model Selection

The three models with the highest cross-validation R² (Gradient Boosting, XGBoost, and Random Forest) were evaluated on the test set:

| **Model**                   | **Train R²** | **Test R²** |   **MAE** |         **MSE** | **Median Abs. Error** |
| --------------------------- | -----------: | ----------: | --------: | --------------: | --------------------: |
| **Random Forest Regressor** |   **0.8936** |  **0.7488** |  4,521.31 |   81,502,270.78 |              2,462.27 |
| XGBoost Regressor           |       0.9609 |      0.7151 |  4,389.53 |   92,444,152.00 |              2,345.41 |
| Gradient Boosting Regressor |       0.9100 |      0.7392 |  4,969.82 |   84,609,846.32 |              2,925.70 |

Gradient Boosting had the highest cross-validation R², but **Random Forest Regressor** was selected as the final model because it achieved the best test R² and the smallest gap between the training and test scores (0.145, compared with 0.171 for Gradient Boosting and 0.246 for XGBoost), which means it overfits the least.

## Hyperparameter Tuning

`GridSearchCV` was used to tune the Random Forest Regressor using **5-Fold Cross-Validation** and **R²** as the scoring metric.

The hyperparameter search included:

* `n_estimators`: 200, 300, 500
* `max_depth`: 8, 10, 12, 15
* `min_samples_split`: 5, 10
* `min_samples_leaf`: 2, 4, 6
* `max_features`: 0.7, sqrt

A total of **144 parameter combinations** were evaluated.

### Best Parameters

```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=0.7,
    random_state=33
)
```

### Best GridSearchCV Score

```text
R² = 0.7175
```

## Final Model Results

The tuned Random Forest Regressor was evaluated on the test set.

| **Metric**            |     **Score** |
| --------------------- | ------------: |
| Train R²              |        0.8936 |
| Test R²               |    **0.7488** |
| MAE                   |      4,521.31 |
| MSE                   | 81,502,270.78 |
| Median Absolute Error |      2,462.27 |

The model achieved a **Test R² of approximately 74.9%** on unseen test data.

The **MAE of 4,521.31** means that the model's predictions differ from the actual car prices by approximately 4,521 on average in absolute terms. The median absolute error (≈ 2,462) is much lower, which suggests that most predictions are close to the real price while a smaller number of expensive or rare cars produce larger errors.

## Actual vs Predicted Prices

An Actual vs Predicted plot was used to visually evaluate how closely the model's predictions follow the actual car prices.

The closer the predictions are to the diagonal reference line, the closer the predicted prices are to the actual prices.

## Limitations

* The Target Encoding of the `Car` feature was fitted on the full dataset before the train/test split, so the reported cross-validation and test scores may be slightly optimistic. Fitting the encoder on the training data only would give a stricter estimate.
* The dataset contains a number of unusually low prices (below 500) that may not reflect real sale prices. They were kept because no consistent pattern was found to justify removing them, and they may limit the achievable accuracy.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* Category Encoders
* Jupyter Notebook

## Machine Learning Concepts

* Exploratory Data Analysis
* Data Cleaning
* Missing Value Handling
* Feature Engineering
* Categorical Encoding
* Target Encoding
* Outlier Analysis
* Train/Test Splitting
* Feature Scaling
* 5-Fold Cross-Validation
* Model Comparison
* GridSearchCV
* Hyperparameter Tuning
* Overfitting Analysis
* Regression Metrics
* Actual vs Predicted Visualization

## Project Structure

```text
car-price-prediction/
├── Car_Price_Prediction.ipynb
├── README.md
└── .gitignore
```

## Author

**Mohamed Khaled Elsayed Ahmed Aboshrief**

Computer Science & Engineering Student
Aspiring Machine Learning Engineer
