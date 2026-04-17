# Diamond Price Prediction

This project predicts diamond prices using simple, interpretable models: **Linear Regression** and **Random Forest**. It's designed for beginners to intermediate learners and follows a clear, serial workflow including data loading, cleaning, EDA, preprocessing, modeling, error analysis, and feature importance.

## Dataset
- Source: Kaggle (diamond_data.csv)
- Target: `price`
- Key features: carat, depth, table, x, y, z, cut, color, clarity
- Rows are dropped if key columns contain nulls.

## Project Structure
- `diamond_price_prediction.ipynb` (or `diamond_price_prediction.py`)
- `diamond_data.csv` (place in the same directory)
- `README.md`

## Requirements
- Python 3.7+
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn

Install dependencies with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
1. Place `diamond_data.csv` in the project folder.
2. Run the notebook or script:
   - Jupyter: open and run all cells
   - Script: `python diamond_price_prediction.py`

## Steps Included
- Import libraries and load data
- Drop rows with nulls in needed columns
- Data overview and EDA (distributions, scatter plots, box plots, correlation heatmap)
- Preprocessing: one-hot encoding, train/test split
- Model building: Linear Regression and Random Forest
- Error analysis: MAE, RMSE, R², residual plots
- Feature importance: Linear Regression coefficients and Random Forest permutation importance
- Key insights and next steps

## Key Insights
- Carat weight is typically the strongest price predictor.
- Cut, color, and clarity show clear price patterns.
- Linear Regression offers transparent coefficients; Random Forest often gives lower error.
- Residual checks and feature correlations help identify modeling opportunities.

## Notes
- Column names in the code match the dataset; update if your CSV uses different names.
- For larger datasets or improved performance, consider regularization or feature engineering (e.g., volume from x, y, z).
