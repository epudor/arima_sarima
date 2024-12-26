Multi-Column Time Series Forecasting with SARIMA

This script forecasts multiple time series data (columns) in an Excel file using SARIMA models.

Requirements:

pandas
matplotlib
pmdarima
statsmodels
scikit-learn
How to Use:

Data Preparation:

Replace "F.xlsx" with the actual filename and path to your Excel file.
Ensure your data has a first column with dates (format compatible with pd.to_datetime).
Run the Script:

Bash

python time_series_forecasting_sarima.py
Output:

The script analyzes each column:

Performs efficient grid search to find optimal SARIMA parameters for the column.
Attempts auto_arima for further parameter refinement.
Uses KFold cross-validation to evaluate model performance.
Visualizes forecasts for the best model on each column.
The script also handles potential errors like missing data files or model convergence issues.
