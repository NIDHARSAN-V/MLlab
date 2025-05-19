import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Sample dataset: Customers with annual income and spending score
data = pd.DataFrame({
'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
'Annual_Income': [15, 16, 17, 18, 19, 80, 85, 90, 95, 100],
'Spending_Score': [39, 81, 6, 77, 40, 60, 50, 88, 80, 42]
}).set_index('id')

print(data)

# Get number of clusters
n_clusters = int(input("No of Clusters: "))

# Apply KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data)

# Show cluster centers
print("\nCluster Centers:\n", kmeans.cluster_centers_)

# Add cluster labels to dataset
data['cluster'] = kmeans.labels_


print(data)
# Print cluster counts
unique, counts = np.unique(kmeans.labels_, return_counts=True)
cluster_counts = dict(zip(unique, counts))
print("Cluster Counts:", cluster_counts)

# Plot clusters
x_col, y_col = data.columns[0], data.columns[1] # Pick first 2 numeric columns
sns.lmplot(
x=x_col,
y=y_col,
data=data,
hue='cluster',
palette='coolwarm',
height=6,
aspect=1,
fit_reg=False
)
plt.title('KMeans Cluster Graph')
plt.show()

# Inertia (how tight the clusters are)
print("KMeans inertia:", kmeans.inertia_)

# Model score (-inertia)
print("KMeans score:", kmeans.score(data.drop(columns=['cluster'])))