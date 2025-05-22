# house_price_prediction.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

print("Loading dataset...")

# Load Boston Housing dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

print("Dataset loaded successfully!\n")
print("First 5 rows of dataset:")
print(data.head())

# ------- New: Label house prices into categories -------
def label_price(value):
    if value < 20:
        return "Low"
    elif value < 35:
        return "Medium"
    else:
        return "High"

data['PriceCategory'] = data['medv'].apply(label_price)

# ------- New: Correlation Heatmap -------
plt.figure(figsize=(10, 8))
sns.heatmap(data.drop(columns=["PriceCategory"]).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")

# -------- Splitting features and target --------
X = data.drop(["medv", "PriceCategory"], axis=1)
y = data["medv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# -------- Model Training --------
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training complete!")

# -------- Model Evaluation --------
y_pred = model.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")

# -------- New: Save Model --------
joblib.dump(model, "house_price_model.pkl")
print("\nTrained model saved as 'house_price_model.pkl'")

# -------- New: Predict on Custom Input --------
sample_input = [[0.03, 0, 2.18, 0, 0.45, 7.2, 45, 6.0, 3, 220, 18.5, 390.0, 5.0]]
custom_prediction = model.predict(sample_input)[0]
print("\nPrediction for custom input:", round(custom_prediction, 2))
