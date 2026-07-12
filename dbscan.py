import pandas as pd
from sklearn.cluster import DBSCAN

# Load Dataset
data = pd.read_csv("fraud_detection.csv")

# Features
X = data[["Amount", "Frequency"]]

# Create Model
model = DBSCAN(
    eps=3000,
    min_samples=3
)

# Train Model
clusters = model.fit_predict(X)

# Add Cluster Column
data["Cluster"] = clusters

print("\nFraud Detection Result\n")
print(data)