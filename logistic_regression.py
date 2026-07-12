import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv("customer_purchase.csv")

# Features and Target
X = data[["Age", "Salary"]]
y = data["Purchased"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LogisticRegression()

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy :", round(accuracy_score(y_test, y_pred), 2))

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# New Customer Prediction
new_customer = [[36, 52000]]

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nCustomer will Purchase the Product")
else:
    print("\nCustomer will NOT Purchase the Product")