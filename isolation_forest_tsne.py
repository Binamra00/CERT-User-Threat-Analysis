import pandas as pd
import pathlib
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuration
base_data_dir = pathlib.Path(r"")
output_dir = base_data_dir / "filtered_output"
feature_input_path = output_dir / "user_features.csv"
tsne_2d_plot_path = output_dir / "tsne_2d_plot.png"
tsne_3d_plot_path = output_dir / "tsne_3d_plot.png"

# Load user features
df = pd.read_csv(feature_input_path)
users = df['user']
zero_variance_cols = ['has_large_attachment_ratio', 'external_email_ratio', 'external_email_count', 'email_frequency_variance',
                      'sensitive_file_ratio', 'external_drive_access_count', 'file_type_diversity', 'risky_domain_count']
features = df.drop(columns=['user'] + zero_variance_cols, errors='ignore')

# Apply Isolation Forest for outlier detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = iso_forest.fit_predict(features)  # -1 for outliers, 1 for inliers

# Apply t-SNE for 2D visualization
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_2d_result = tsne_2d.fit_transform(features)

# Plot 2D t-SNE
plt.figure(figsize=(10, 6))
colors = {1: 'blue', -1: 'red'}
for label in colors:
    mask = df['anomaly'] == label
    plt.scatter(tsne_2d_result[mask, 0], tsne_2d_result[mask, 1], c=colors[label], label='Inlier' if label == 1 else 'Outlier', alpha=0.6)
plt.title('t-SNE 2D Visualization with Isolation Forest Outliers')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.savefig(tsne_2d_plot_path)
plt.close()

# Apply t-SNE for 3D visualization
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
tsne_3d_result = tsne_3d.fit_transform(features)

# Plot 3D t-SNE
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for label in colors:
    mask = df['anomaly'] == label
    ax.scatter(tsne_3d_result[mask, 0], tsne_3d_result[mask, 1], tsne_3d_result[mask, 2], c=colors[label], label='Inlier' if label == 1 else 'Outlier', alpha=0.6)
ax.set_title('t-SNE 3D Visualization with Isolation Forest Outliers')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
ax.legend()
plt.savefig(tsne_3d_plot_path)
plt.close()