import pandas as pd
import pathlib
from sklearn.ensemble import IsolationForest

# Configuration
base_data_dir = pathlib.Path(r"")
output_dir = base_data_dir / "filtered_output"
feature_input_path = output_dir / "user_features.csv"
outlier_output_path = output_dir / "isolation_forest_outliers.csv"

# Load user features
df = pd.read_csv(feature_input_path)
users = df['user']
zero_variance_cols = ['has_large_attachment_ratio', 'external_email_ratio', 'external_email_count', 'email_frequency_variance',
                      'sensitive_file_ratio', 'external_drive_access_count', 'file_type_diversity', 'risky_domain_count']
features = df.drop(columns=['user'] + zero_variance_cols, errors='ignore')

# Apply Isolation Forest for outlier detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['is_outlier'] = iso_forest.fit_predict(features)  # -1 for outliers, 1 for inliers

# Save outlier labels
outlier_df = df[['user', 'is_outlier']]
outlier_df.to_csv(outlier_output_path, index=False)