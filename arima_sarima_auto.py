import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import itertools

# Function to perform efficient grid search with early stopping
def efficient_grid_search(data, param_grid, metric):
    best_aic = float("inf")
    best_params = None
    for params in param_grid:
        order = params[:3]
        seasonal_order = params[3:]
        try:
            model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
            result = model.fit(disp=False)
            if not np.isnan(metric(result)):  # Check for valid metric value
                current_metric = metric(result)
                if current_metric < best_aic:
                    best_aic = current_metric
                    best_params = params
                    # Early stopping if improvement is minimal (adjust threshold as needed)
                    if best_aic - current_metric < 0.01:
                        break
        except:
            continue
    return best_params

file = "F.xlsx"
# Load your data
try:
    # Load data and specify the first column as the index with date parsing
    data = pd.read_excel(f"C:\\Users\\z\\Desktop\\Real Estate\\Promora\\Datos\\csv\\{file}", parse_dates=[0], index_col=0)
    # Sort the DataFrame by the index (dates)
    data.sort_index(inplace=True)
except FileNotFoundError:
    print(f"Error: File {file} not found. Please check the filename and path.")
    exit()

# Define parameter ranges (consider reducing m range)
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)
P_values = range(0, 2)  # Adjust based on seasonality strength
D_values = range(0, 2)  # Adjust based on seasonality strength
Q_values = range(0, 2)
m_values = [1, 2, 3, 5, 7, 8, 12, 13, 21, 30, 34]  # Explore a smaller seasonal period range

# Create a more focused parameter grid using itertools product
param_grid = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, m_values))

# Define metric function (replace with your preferred metric)
def aic_metric(result):
    return result.aic

# Analyze each column
for column in data.columns:
    print(f"Analyzing {column}")

    # Perform efficient grid search
    best_params = efficient_grid_search(data[column], param_grid, aic_metric)

    # Print results and handle non-convergence
    if best_params is not None:
        print("Best Parameters for", column, ":", best_params)
        print("Best Metric for", column, ":", aic_metric(SARIMAX(data[column], order=best_params[:3], seasonal_order=best_params[3:]).fit(disp=False)))
    else:
        print("No convergence found for", column, ". Unable to determine best parameters.")

    # Perform auto_arima for further refinement (optional)
    try:
        best_model = auto_arima(data[column], start_p=p_values[0], start_q=q_values[0], max_p=max(p_values), max_q=max(q_values), m=12, seasonal=False, suppress_warnings=True, stepwise=True)
        order = best_model.order
    except:
        print(f"Auto ARIMA failed for {column}. Using grid search parameters.")
        order = best_params[:3]  # Fallback to grid search results

    # KFold cross-validation for more robust evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(data[column]):
        train, test = data[column].iloc[train_index], data[column].iloc[test_index]
        try:
            model = SARIMAX(train, order=order, seasonal_order=best_params[3:])
            fitted_model = model.fit()

            # Forecast future values
            forecast = fitted_model.forecast(steps=len(test))  # Forecast the same number of steps as the test data

            # Plot the forecasts for the best model
            plt.plot(train.index, train.values, label='Training Data')
            plt.plot(test.index, test.values, label='Testing Data')
            plt.plot(test.index, forecast.values, label='Forecast')
            plt.title(f'Best Forecast for {column}')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

        except Exception as e:
            print(f"An error occurred during model fitting or forecasting for {column}: {e}")




