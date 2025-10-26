# --- Imports ---
import pandas as pd
import pathlib
from sklearn.cluster import KMeans
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import re # Import regular expressions for cleaning names

# --- Configuration ---
base_data_dir = pathlib.Path(r"")
output_dir = base_data_dir / "filtered_output"
feature_input_path = output_dir / "user_features.csv"
kmeans_outliers_output_path = output_dir / "kmeans_outliers.csv" # Output path for the K-Means outliers file
anova_results_output_path = output_dir / "univariate_anova_results.csv" # New output path for ANOVA results

# --- Load Data ---
try:
    df = pd.read_csv(feature_input_path)
    print(f"Successfully loaded data from {feature_input_path}")
except FileNotFoundError:
    print(f"Error: Data file not found at {feature_input_path}")
    print("Please ensure the file exists and the path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    exit()

# --- Define zero_variance_cols and other columns to exclude from features ---
# This list is used both for feature selection for clustering/ANOVA and plotting exclusion
# Added common non-feature columns including potential outlier flags from previous steps
cols_to_exclude_from_features = [
    'user',
    'cluster', # Exclude the target variable itself
    'kmeans_outlier', # Exclude if this was loaded or created
    'is_kmeans_outlier', # Exclude if this was loaded or created
    'isolation_forest_anomaly', # Exclude if this was created
    # Your specific zero variance columns
    'has_large_attachment_ratio',
    'external_email_ratio',
    'external_email_count',
    'email_frequency_variance',
    'sensitive_file_ratio',
    'external_drive_access_count',
    'file_type_diversity',
    'risky_domain_count'
]

# --- Prepare Features for Clustering/Analysis ---
# Create a DataFrame containing only the features we want to use
# Use errors='ignore' in drop in case some columns in cols_to_exclude_from_features don't exist
features = df.drop(columns=cols_to_exclude_from_features, errors='ignore')

# Get the list of actual feature column names that remain after dropping
feature_cols = list(features.columns)

print(f"Identified {len(feature_cols)} features for clustering and ANOVA.")

# --- Perform K-Means Clustering ---
print("\nPerforming K-Means Clustering...")
optimal_clusters = 4 # Use your determined optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)

# *** This line adds the 'cluster' column to the 'df' DataFrame currently in memory ***
# We fit on the 'features' DataFrame, but the result is added to the original 'df'
df['cluster'] = kmeans.fit_predict(features)

print("K-Means clustering complete. 'cluster' column added to DataFrame.")

# Optional: Save K-Means outliers (from your previous K-Means script)
cluster_sizes = df['cluster'].value_counts()
if not cluster_sizes.empty:
    # Assuming the smallest cluster is index 3 based on your output
    # It's safer to use idxmin() in case the smallest cluster label changes
    smallest_cluster_label = cluster_sizes.idxmin()
    outliers_df = df[df['cluster'] == smallest_cluster_label][['user', 'cluster']]
    outliers_df.to_csv(kmeans_outliers_output_path, index=False)
    print(f"K-Means outliers (from smallest cluster {smallest_cluster_label}) saved to {kmeans_outliers_output_path}")
else:
     print("No clusters found to identify smallest cluster/outliers.")


# --- Data Preparation for ANOVA: Clean Column Names ---
print("\nPreparing data for ANOVA by cleaning feature column names...")

# Create a copy of the DataFrame to clean names without affecting the original df
# for potential later use if needed, though here we just need it for ANOVA
df_anova = df[['cluster'] + feature_cols].copy() # Select only cluster and features for ANOVA

# Function to clean column names for statsmodels formulas
def clean_col_name(col_name):
    # Replace problematic characters with underscores
    cleaned = col_name.replace(' - ', '_').replace('-', '_').replace(' ', '_').replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_')
    # Remove any characters that are not alphanumeric or underscore
    cleaned = re.sub(r'[^\w]', '', cleaned)
    # Ensure it doesn't start with a number (statsmodels formula rule) - unlikely with your names but good practice
    if re.match(r'^\d', cleaned):
        cleaned = '_' + cleaned
    # Ensure it doesn't start or end with an underscore
    cleaned = cleaned.strip('_')
    return cleaned

# Create a dictionary mapping original feature names to cleaned names
cleaned_feature_name_map = {col: clean_col_name(col) for col in feature_cols}

# Rename columns in the df_anova DataFrame
df_anova.rename(columns=cleaned_feature_name_map, inplace=True)

# Update the list of feature columns to use the *cleaned* names for the ANOVA loop
cleaned_feature_cols = list(df_anova.drop(columns=['cluster']).columns)

print(f"Cleaned names for {len(cleaned_feature_cols)} features.")

# --- Perform Univariate ANOVA ---
# The 'df_anova' DataFrame now has cleaned column names and the 'cluster' column.

print(f"\nPerforming ANOVA for {len(cleaned_feature_cols)} features comparing means across clusters...")

# Dictionary to store ANOVA results
anova_results_data = []

# Ensure the cluster column is treated as a categorical variable for ANOVA
df_anova['cluster'] = df_anova['cluster'].astype('category')

# Perform ANOVA for each feature using the cleaned feature names
for feature in cleaned_feature_cols:
    # Check if the feature exists and is numeric (should be if cleaned_feature_cols is correct, but double-check)
    if feature not in df_anova.columns or not pd.api.types.is_numeric_dtype(df_anova[feature]):
        print(f"Skipping ANOVA for '{feature}' (not found or not numeric).")
        anova_results_data.append({
            'Feature': feature,
            'P_value': np.nan,
            'F_value': np.nan,
            'Error': 'Not found or not numeric'
        })
        continue

    try:
        # Build the ANOVA formula using the cleaned feature name
        formula = f'{feature} ~ C(cluster)'

        # Fit the OLS model and perform ANOVA
        model = ols(formula, data=df_anova).fit()
        anova_table = sm.stats.anova_lm(model, typ=2) # Use type 2 ANOVA

        # Extract the p-value and F-value for the cluster variable
        # The index name should consistently be 'C(cluster)' now if formula uses C()
        p_value = np.nan # Initialize with NaN
        f_value = np.nan # Initialize with NaN
        error_msg = None

        if 'C(cluster)' in anova_table.index:
             p_value = anova_table.loc['C(cluster)', 'PR(>F)']
             f_value = anova_table.loc['C(cluster)', 'F']
        elif 'cluster' in anova_table.index: # Fallback (less common with C() in formula)
             p_value = anova_table.loc['cluster', 'PR(>F)']
             f_value = anova_table.loc['cluster', 'F']
        else:
             error_msg = "Cluster term not found in ANOVA table"
             print(f"Warning: {error_msg} for {feature}")


        anova_results_data.append({
            'Feature': feature,
            'P_value': p_value,
            'F_value': f_value,
            'Error': error_msg
        })

    except Exception as e:
        # Catch potential errors during ANOVA (e.g., zero variance within groups, singular matrix)
        error_msg = str(e)
        print(f"Error performing ANOVA for feature '{feature}': {error_msg}")
        anova_results_data.append({
            'Feature': feature,
            'P_value': np.nan,
            'F_value': np.nan,
            'Error': error_msg
        })


# --- Process and Save ANOVA Results ---

# Convert results list to a DataFrame
anova_results_df = pd.DataFrame(anova_results_data)

# Define significance level
alpha = 0.05

# Add 'Significance_Level' column
# Use np.select for multiple conditions (NaN, Significant, Less Significant)
conditions = [
    anova_results_df['P_value'].isna(), # Condition 1: P_value is NaN
    anova_results_df['P_value'] < alpha # Condition 2: P_value is less than alpha
]
choices = [
    'Skipped/Error',         # Choice 1: If P_value is NaN
    'Significant'            # Choice 2: If P_value < alpha
]
# Default choice: if none of the above conditions are met
anova_results_df['Significance_Level'] = np.select(conditions, choices, default='Less Significant')


# Sort the results by P_value (ascending)
anova_results_df = anova_results_df.sort_values(by='P_value', ascending=True)

# Select and reorder columns for the final output CSV
output_columns = ['Feature', 'P_value', 'F_value', 'Significance_Level', 'Error']
anova_results_df = anova_results_df[output_columns]


# Save the results to a CSV file
try:
    anova_results_df.to_csv(anova_results_output_path, index=False)
    print(f"\nANOVA results saved to {anova_results_output_path}")
except Exception as e:
    print(f"\nError saving ANOVA results to CSV: {e}")


# --- Display Summary (Optional) ---
print("\n--- ANOVA Results Summary (Top features by P-value) ---")
# Display features that are 'Significant' or have a p-value close to 0
significant_summary = anova_results_df[anova_results_df['Significance_Level'] == 'Significant'].head(20) # Display top 20 significant
if not significant_summary.empty:
    print("\nFeatures with statistically significant difference across clusters (p < 0.05):")
    print(significant_summary[['Feature', 'P_value', 'F_value']].to_string(index=False, formatters={'P_value': '{:.4f}'.format, 'F_value': '{:.2f}'.format}))
else:
    print("\nNo features found with statistically significant differences (p < 0.05).")

errored_features = anova_results_df[anova_results_df['Significance_Level'] == 'Skipped/Error']
if not errored_features.empty:
    print(f"\n--- {len(errored_features)} Features Skipped or Encountered Errors during ANOVA ---")
    # Print just the feature name and error message for skipped/errored features
    print(errored_features[['Feature', 'Error']].to_string(index=False))


print("\nScript finished.")