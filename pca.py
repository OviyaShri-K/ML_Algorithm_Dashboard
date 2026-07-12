import pandas as pd
from sklearn.decomposition import PCA

# Load Dataset
data = pd.read_csv("student_performance.csv")

# Features
X = data[["Maths", "Science", "English", "Computer"]]

# Create PCA Model
model = PCA(n_components=2)

# Transform Data
X_pca = model.fit_transform(X)

print("PCA Result\n")
print(X_pca)