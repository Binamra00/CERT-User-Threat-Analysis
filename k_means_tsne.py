import pandas as pd
import pathlib
from sklearn.ensemble import IsolationForest # Keep if you still want IF results calculated
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np # Needed for np.where

# Configuration
base_data_dir = pathlib.Path(r"")
output_dir = base_data_dir / "filtered_output"
feature_input_path = output_dir / "user_features.csv"
kmeans_outliers_path = output_dir / "kmeans_outliers.csv" # Path to the K-Means outliers file
tsne_2d_plot_path = output_dir / "tsne_2d_plot_kmeans_outliers.png" # Change output name for clarity
tsne_3d_plot_path = output_dir / "tsne_3d_plot_kmeans_outliers.png" # Change output name for clarity

# Load user features
df = pd.read_csv(feature_input_path)
# Keep users separate if needed later, but we'll use the 'user' column in df
# users = df['user'] # Not strictly necessary for this modification

zero_variance_cols = ['has_large_attachment_ratio', 'external_email_ratio', 'external_email_count', 'email_frequency_variance',
                      'sensitive_file_ratio', 'external_drive_access_count', 'file_type_diversity', 'risky_domain_count']
features = df.drop(columns=['user'] + zero_variance_cols, errors='ignore')

# --- NEW: Load K-Means Outliers and Flag Main DataFrame ---
try:
    kmeans_outliers_df = pd.read_csv(kmeans_outliers_path)
    # Get the list of user IDs identified as outliers by K-Means
    kmeans_outlier_users = set(kmeans_outliers_df['user'].unique()) # Use a set for efficient lookup

    # Create a new column in the main df: -1 if user is in the K-Means outlier list, 1 otherwise
    df['is_kmeans_outlier'] = np.where(df['user'].isin(kmeans_outlier_users), -1, 1)

    print(f"Loaded {len(kmeans_outlier_users)} K-Means outliers from {kmeans_outliers_path}")

except FileNotFoundError:
    print(f"Error: K-Means outliers file not found at {kmeans_outliers_path}")
    print("Please ensure 'kmeans_outliers.csv' exists before running this code.")
    exit() # Exit or handle the error appropriately

# --- Isolation Forest (Optional - Keep if you still want IF results calculated, but not for plotting colors) ---
# iso_forest = IsolationForest(contamination=0.1, random_state=42)
# df['isolation_forest_anomaly'] = iso_forest.fit_predict(features) # Save IF results to a different column name

# --- Apply t-SNE for 2D visualization ---
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_2d_result = tsne_2d.fit_transform(features)

# --- Plot 2D t-SNE - MODIFIED TO USE 'is_kmeans_outlier' ---
plt.figure(figsize=(10, 6))
# Use the same colors dictionary, assuming -1 for outlier and 1 for inlier
colors = {1: 'blue', -1: 'red'}
# Iterate through the labels defined by your new K-Means outlier column
for label in colors:
    # *** Use the 'is_kmeans_outlier' column for the mask ***
    mask = df['is_kmeans_outlier'] == label
    plt.scatter(tsne_2d_result[mask, 0], tsne_2d_result[mask, 1], c=colors[label], label='Not K-Means Outlier' if label == 1 else 'K-Means Outlier', alpha=0.6)

plt.title('t-SNE 2D Visualization with K-Means Outliers') # Update title
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.savefig(tsne_2d_plot_path)
plt.close()

# --- Apply t-SNE for 3D visualization ---
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
tsne_3d_result = tsne_3d.fit_transform(features)

# --- Plot 3D t-SNE - MODIFIED TO USE 'is_kmeans_outlier' ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Use the same colors dictionary
for label in colors:
    # *** Use the 'is_kmeans_outlier' column for the mask ***
    mask = df['is_kmeans_outlier'] == label
    ax.scatter(tsne_3d_result[mask, 0], tsne_3d_result[mask, 1], tsne_3d_result[mask, 2], c=colors[label], label='Not K-Means Outlier' if label == 1 else 'K-Means Outlier', alpha=0.6)

ax.set_title('t-SNE 3D Visualization with K-Means Outliers') # Update title
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
ax.legend()
plt.savefig(tsne_3d_plot_path)
plt.close()

print("t-SNE plots with K-Means outliers saved.")