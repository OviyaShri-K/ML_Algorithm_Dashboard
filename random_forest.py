import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("employee_attrition.csv")

# Features and Target
X = data[["Age", "Experience", "Salary"]]
y = data["Attrition"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy :", round(accuracy_score(y_test, y_pred), 2))

# New Employee Prediction
new_employee = [[27, 3, 33000]]

prediction = model.predict(new_employee)

if prediction[0] == 1:
    print("Employee Likely to Leave")
else:
    print("Employee Likely to Stay")