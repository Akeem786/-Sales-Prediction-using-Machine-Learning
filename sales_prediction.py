# 📊 Sales Prediction Using Machine Learning

# ✅ Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Step 2: Load Dataset
data = pd.read_csv("sales_data.csv")
data.head()

# ✅ Step 3: EDA (Exploratory Data Analysis)
print(data.info())
print(data.describe())

# Check missing values
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# ✅ Step 4: Data Preprocessing
# Drop rows with null values (or use imputation)
data = data.dropna()

# Encode categorical variables if any
# Example:
# data['Category'] = pd.get_dummies(data['Category'], drop_first=True)

# ✅ Step 5: Feature Selection
X = data.drop("Sales", axis=1)
y = data["Sales"]

# Convert all columns to numeric if needed
X = pd.get_dummies(X, drop_first=True)

# ✅ Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 7: Train Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    results[name] = {"RMSE": rmse, "R2": r2}

# ✅ Step 8: Results
result_df = pd.DataFrame(results).T
print(result_df)

# ✅ Step 9: Visualization
result_df.plot(kind="bar", figsize=(10, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ✅ Step 10: Conclusion
# You can print best model or save it using joblib or pickle
