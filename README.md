# Car Price Prediction

A machine learning regression project for predicting used car prices based on vehicle characteristics such as production year, mileage, engine volume, fuel type, gearbox, drive wheels, and other features.

The project covers an end-to-end machine learning workflow, including data preprocessing, feature engineering, categorical encoding, model comparison using cross-validation, hyperparameter tuning, and final model evaluation.

## Project Overview

The goal of this project is to build a regression model capable of predicting the price of a used car based on its available features.

## Data Preprocessing

The project includes several data preprocessing and feature engineering steps:

* Handling missing values
* Cleaning and converting numerical features
* Handling categorical features
* Handling outliers
* Combining car manufacturer and model information
* Target Encoding for the high-cardinality `Car` feature
* Removing an extreme price outlier

## Model Comparison

Eight regression models were evaluated using **5-Fold Cross-Validation** with **R²** as the evaluation metric:

* Linear Regression
* SGD Regressor
* Lasso Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Decision Tree Regressor
* K-Nearest Neighbors Regressor

### Cross-Validation Results

| Model                         | Mean CV R² |
| ----------------------------- | ---------: |
| Linear Regression             |     0.2623 |
| SGD Regressor                 |  -2.49e+29 |
| Lasso Regression              |     0.2624 |
| Ridge Regression              |     0.2624 |
| **Random Forest Regressor**   | **0.7447** |
| Gradient Boosting Regressor   |     0.6531 |
| Decision Tree Regressor       |     0.6020 |
| K-Nearest Neighbors Regressor |     0.3587 |

Based on the cross-validation results, **Random Forest Regressor** achieved the highest mean R² and was selected for further hyperparameter tuning.

## Hyperparameter Tuning

`GridSearchCV` was used to tune the Random Forest Regressor using **5-Fold Cross-Validation** and **R²** as the scoring metric.

The hyperparameter search included:

* `n_estimators`: 300, 500
* `max_depth`: 8, 10, 15
* `min_samples_split`: 2, 5
* `min_samples_leaf`: 1, 2
* `max_features`: 0.8

A total of **24 parameter combinations** were evaluated.

### Best Parameters

```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.8,
    random_state=33
)
```

### Best GridSearchCV Score

```text
R² = 0.7431
```

## Final Model Results

The tuned Random Forest Regressor was evaluated on the test set.

| Metric                |         Score |
| --------------------- | ------------: |
| Train R²              |        0.9394 |
| Test R²               |    **0.7523** |
| MAE                   |      4,467.51 |
| MSE                   | 80,376,900.24 |
| Median Absolute Error |      2,440.52 |

The model achieved a **Test R² of approximately 75.2%** on unseen test data.

The **MAE of 4,467.51** means that the model's predictions differ from the actual car prices by approximately 4,468 on average in absolute terms.

## Actual vs Predicted Prices

An Actual vs Predicted plot was used to visually evaluate how closely the model's predictions follow the actual car prices.

The closer the predictions are to the diagonal reference line, the closer the predicted prices are to the actual prices.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
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
* 5-Fold Cross-Validation
* Model Comparison
* GridSearchCV
* Hyperparameter Tuning
* Regression Metrics
* Actual vs Predicted Visualization

## Project Structure

```text
car-price-prediction/
├── car_price_prediction.ipynb
├── README.md
└── .gitignore
```

## Author

**Mohamed Khaled Elsayed Ahmed Aboshrief**

Computer Science & Engineering Student
Aspiring Machine Learning Engineer
