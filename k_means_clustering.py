import pandas as pd
import pathlib
from sklearn.cluster import KMeans
import numpy as np

# Configuration
base_data_dir = pathlib.Path(r"")
output_dir = base_data_dir / "filtered_output"
feature_input_path = output_dir / "user_features.csv"
outliers_output_path = output_dir / "kmeans_outliers.csv"

# Load user features
df = pd.read_csv(feature_input_path)

# Exclude 'user' column and zero-variance columns
zero_variance_cols = ['has_large_attachment_ratio', 'external_email_ratio', 'external_email_count', 'email_frequency_variance',
                      'sensitive_file_ratio', 'external_drive_access_count', 'file_type_diversity', 'risky_domain_count']
features = df.drop(columns=['user'] + zero_variance_cols, errors='ignore')

# Apply K-Means clustering with optimal number of clusters (assumed 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features)

# Identify outliers as users in the smallest cluster(s)
cluster_sizes = df['cluster'].value_counts()
smallest_cluster = cluster_sizes.idxmin()
outliers = df[df['cluster'] == smallest_cluster][['user', 'cluster']]

# Save outliers to CSV
outliers.to_csv(outliers_output_path, index=False)