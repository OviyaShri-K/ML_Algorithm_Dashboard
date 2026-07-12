import pandas as pd
from sklearn.cluster import KMeans

# Load Dataset
data = pd.read_csv("customer_segmentation.csv")

# Features
X = data[["Annual_Income", "Spending_Score"]]

# Create Model
model = KMeans(
    n_clusters=3,
    random_state=42
)

# Train Model
model.fit(X)

# Cluster Labels
clusters = model.labels_

# Add Cluster Column
data["Cluster"] = clusters

print(data)