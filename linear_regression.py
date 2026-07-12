import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("students.csv")

# Features and Target
X = data[["Hours_Studied"]]
y = data["Marks"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict Test Data
y_pred = model.predict(X_test)

# Model Parameters
print("Intercept :", model.intercept_)
print("Coefficient :", model.coef_[0])

# Evaluation Metrics
print("\nModel Performance")
print("MAE :", round(mean_absolute_error(y_test, y_pred), 2))
print("MSE :", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score :", round(r2_score(y_test, y_pred), 2))

# New Prediction
hours = [[7.5]]
predicted_marks = model.predict(hours)

print(f"\nPredicted Marks for {hours[0][0]} hours of study: {predicted_marks[0]:.2f}")