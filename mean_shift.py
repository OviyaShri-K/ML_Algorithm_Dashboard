import pandas as pd
from sklearn.cluster import MeanShift

# Load Dataset
data = pd.read_csv("store_customers.csv")

# Features
X = data[["Monthly_Spending", "Visits_Per_Month"]]

# Create Model
model = MeanShift()

# Train Model
model.fit(X)

# Cluster Labels
clusters = model.labels_

print("Cluster Labels:")
print(clusters)

# Cluster Centers
print("\nCluster Centers:")
print(model.cluster_centers_)