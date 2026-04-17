import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load dataset
df = pd.read_csv('diamond_data.csv')

# Display first few rows
print(df.head())

# Keep only necessary columns
df = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']]

# Drop any rows with missing values
df = df.dropna()

print(f"Dataset shape: {df.shape}")

# Distribution of target variable (price)
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True, bins=30)
plt.title('Distribution of Diamond Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig('price_distribution.png')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()

# Categorical features count
plt.figure(figsize=(12, 4))
for i, col in enumerate(['cut', 'color', 'clarity']):
    plt.subplot(1, 3, i+1)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.savefig('categorical_distributions.png')
plt.show()

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Identify categorical and numerical columns
categorical_cols = ['cut', 'color', 'clarity']
numerical_cols = ['carat', 'depth', 'table']

#Preprocessor: One-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # keep numerical columns as-is
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
lr_model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())])

# Random Forest model
rf_model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")
    
    # Actual vs Predicted plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.savefig(f'{model_name.replace(" ", "_")}_actual_vs_predicted.png')
    plt.show()
    
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Residual Plot')
    plt.savefig(f'{model_name.replace(" ", "_")}_residual_plot.png')
    plt.show()

# Evaluate both models
evaluate_model(lr_model, X_test, y_test, "Linear Regression")
evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Extract feature names after one-hot encoding
preprocessor.fit(X)
cat_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_cols))
feature_names = cat_feature_names + numerical_cols

# Get importances
importances = rf_model.named_steps['regressor'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\nFeature Importances (Random Forest):")
print(feat_imp)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance_random_forest.png')
plt.show()