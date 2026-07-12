import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Load Dataset
data = pd.read_csv("mall_customers.csv")

# Features
X = data[["Annual_Income", "Spending_Score"]]

# Create Model
model = AgglomerativeClustering(
    n_clusters=3
)

# Train Model
clusters = model.fit_predict(X)

# Add Cluster Column
data["Cluster"] = clusters

print("\nCustomer Segmentation Result\n")
print(data)