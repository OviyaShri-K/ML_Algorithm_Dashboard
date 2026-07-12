import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("mobile_recommendation.csv")

# Features and Target
X = data[["Price", "RAM"]]
y = data["Category"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = KNeighborsClassifier(n_neighbors=5)

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy :", round(accuracy_score(y_test, y_pred), 2))

# New Mobile Prediction
new_mobile = [[27000, 8]]

prediction = model.predict(new_mobile)

categories = {
    0: "Budget Phone",
    1: "Mid-Range Phone",
    2: "Premium Phone"
}

print("Recommended Category :", categories[prediction[0]])