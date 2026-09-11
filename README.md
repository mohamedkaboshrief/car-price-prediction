# Car Price Prediction

A machine learning project for predicting used car prices using data preprocessing, feature engineering, and regression models.

## Project Overview

The goal of this project is to build a machine learning model that predicts the price of a used car based on its characteristics.

The dataset contains information about cars, including production year, mileage, engine volume, fuel type, gearbox type, drive wheels, and other features.

## Data Preprocessing

The project includes several data preprocessing and feature engineering steps:

* Handling missing values
* Cleaning and converting numerical features
* Handling categorical features
* Feature engineering
* Combining car manufacturer and model information
* Target Encoding for the high-cardinality `Car` feature
* Removing an extreme price outlier

## Models

Several regression models were tested:

* Linear Regression
* Lasso Regression
* Ridge Regression
* Decision Tree Regressor
* K-Nearest Neighbors Regressor
* Support Vector Regression
* Gradient Boosting Regressor
* Random Forest Regressor

The **Random Forest Regressor** achieved the best performance among the tested models.

## Hyperparameter Tuning

The Random Forest model was improved by adjusting hyperparameters such as:

* `n_estimators`
* `max_depth`
* `random_state`

The selected model configuration was:

```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    random_state=33
)
```

## Model Results

The final Random Forest Regressor achieved:

| Metric                |         Score |
| --------------------- | ------------: |
| Train R²              |        0.8607 |
| Test R²               |        0.6979 |
| MAE                   |      5,584.31 |
| MSE                   | 98,022,220.97 |
| Median Absolute Error |      3,409.62 |

The model achieved a **Test R² of approximately 69.8%** on unseen test data.

## Technologies

* Python
* Pandas
* Scikit-learn
* Category Encoders
* Jupyter Notebook

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

