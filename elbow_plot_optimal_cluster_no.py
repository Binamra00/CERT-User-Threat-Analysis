import pandas as pd
import pathlib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Configuration
base_data_dir = pathlib.Path(r"")
output_dir = base_data_dir / "filtered_output"
feature_input_path = output_dir / "user_features.csv"
elbow_plot_path = output_dir / "elbow_plot.png"

# Load user features
df = pd.read_csv(feature_input_path)

# Exclude 'user' column and zero-variance columns identified previously
zero_variance_cols = ['has_large_attachment_ratio', 'external_email_ratio', 'external_email_count', 'email_frequency_variance',
                      'sensitive_file_ratio', 'external_drive_access_count', 'file_type_diversity', 'risky_domain_count']
features = df.drop(columns=['user'] + zero_variance_cols, errors='ignore')

# Compute WCSS for different numbers of clusters (1 to 10)
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.savefig(elbow_plot_path)
plt.close()