
**PROJECT: CERT User Threat Analysis**


## 1. Project Overview

This project implements a complete User Behavior Analysis (UBA) pipeline to detect potential insider threats using the CERT r4.2 dataset. Given the massive size of the dataset (15.1 GB) and typical hardware constraints (e.g., 16 GB RAM), the primary challenge is to perform a statistically sound analysis.

The workflow consists of four main phases:
1.  **Data Filtering & Downsampling:** Drastically reducing the dataset size using a session-based, stratified sampling methodology.
2.  **Feature Engineering:** Transforming the filtered logs into a user-level feature matrix by calculating 76 statistical features per user.
3.  **Unsupervised Modeling:** Applying K-Means clustering and Isolation Forest anomaly detection to identify anomalous users.
4.  **Statistical Validation & Visualization:** Using ANOVA to determine which features best separate the user clusters and t-SNE to visualize the results.

---

## 2. Setup & Configuration

Before running the analysis, you must set up your environment and download the dataset.

### Dependencies
This project requires Python 3 and the following libraries:
* pandas
* dask
* scikit-learn (sklearn)
* matplotlib
* statsmodels
* numpy
* tqdm (used in `downsampling_http_csv.py`)

You can install them using pip:
`pip install pandas dask scikit-learn matplotlib statsmodels numpy tqdm`

### Data Download
1.  Download the **CERT Insider Threat r4.2 dataset**. This dataset is publicly available on platforms like Kaggle.
2.  Unzip the dataset into a local directory. Your directory structure should look something like this:
```
/path/to/your/dataset/r4.2/
|-- email.csv
|-- http.csv
|-- file.csv
|-- device.csv
|-- logon.csv
|-- psychometric.csv
|-- LDAP/
|   |-- 2009-12.csv
|   |-- 2010-01.csv
|   |-- ...etc
```
### **IMPORTANT: Path Configuration**
You must manually set the base path to your dataset in all the Python scripts.

1.  Open each `.py` script (e.g., `downsampling.py`, `feature_engineering.py`, etc.).
2.  Find this line near the top of each file:
    ```python
    base_data_dir = pathlib.Path(r"")
    ```
3.  Change the empty string `r""` to the full, absolute path of your `r4.2` folder. **Use a raw string literal (r"...") to avoid issues with backslashes.**

**Example:**
If your data is at `C:\Users\YourName\Desktop\r4.2`, you would change the line to:

base_data_dir = pathlib.Path(r"C:\Users\YourName\Desktop\r4.2")
---

## 3. Methodology & Workflow

### Step 1: Data Filtering & Downsampling
* **Scripts:** `downsampling.py`, `downsampling_http_csv.py`
* **Report:** `Step1_Filtering_and_DownSampling_Report.html`

The process begins by filtering the 12,000 LDAP entries to 1,000 unique users. The core of the downsampling relies on sessionizing the `logon.csv` data.

Due to memory constraints, an ideal statistical sample (e.g., 385 sessions/user) was not feasible. Instead, a **stratified temporal sampling** approach was adopted:
1.  The project timeline was divided into 6 equal time bins.
2.  **6 sessions** were randomly sampled per user (one from each bin) to ensure representative temporal coverage. This resulted in 6,000 total sampled sessions.
3.  All other activity logs (`email.csv`, `file.csv`, `device.csv`, `http.csv`) were then filtered to retain **only** the events that occurred within these 6,000 sampled session windows.
4.  A separate script (`downsampling_http_csv.py`) was used to process the massive `http.csv` (28.4M rows) in batches to prevent memory crashes.

**Result:** The raw activity logs were reduced by ~98.5%, creating a manageable set of `filtered_*.csv` files.

### Step 2: Feature Engineering
* **Script:** `feature_engineering.py`
* **Report:** `Step2_Feature_Engineering_Report.html`

This script aggregates all `filtered_*.csv` files into a single `user_features.csv` file (1,000 rows, one per user).
1.  A **variance-based heuristic** (`k = ceil(log2(rows) * 0.8)`) was used to determine a target number of features to extract from each data source, resulting in a target of **76 features**.
2.  Features are statistical aggregations, including means, variances, counts, ratios, and interaction terms (e.g., `session_duration_mean`, `after_hours_email_ratio`, `file_type_diversity`, `neuroticism_external_access_interaction`).
3.  The final data is scaled (StandardScaler) and one-hot encoded, preparing it for machine learning.

### Step 3: Modeling & Outlier Detection
* **Scripts:** `elbow_plot_optimal_cluster_no.py`, `k_means_clustering.py`, `outlier_analysis.py`
* **Report:** `Step3_Clustering_Report.html`

Two unsupervised methods are used to identify outliers:
1.  **K-Means Clustering:**
    * `elbow_plot_optimal_cluster_no.py` is run to determine the optimal number of clusters. The Elbow plot (`elbow_plot.png`) identifies **k=4** as the optimal number.
    * `k_means_clustering.py` applies K-Means with k=4. The smallest cluster is flagged as the K-Means outlier group.
2.  **Isolation Forest:**
    * `outlier_analysis.py` applies the Isolation Forest algorithm with a `contamination` parameter of 0.1 (10%), identifying 100 users as outliers.

### Step 4: Statistical Validation & Visualization
* **Scripts:** `univariate_anova_test.py`, `k_means_tsne.py`, `isolation_forest_tsne.py`
* **Reports:** `Step4_ANOVA_Test_Report.html`, `Final_Report.html`

1.  **ANOVA Test:** `univariate_anova_test.py` performs a one-way ANOVA test for each of the 76 features against the 4 K-Means cluster labels. This identifies which features (e.g., `external_access_count`) are the most statistically significant differentiators between the behavioral groups.
2.  **t-SNE Visualization:** The `..._tsne.py` scripts are used to reduce the high-dimensional feature space into 2D and 3D plots, coloring the users by their outlier status (from both K-Means and Isolation Forest) to visually inspect cluster separation and anomalies.

---

## 4. Key Findings
* **Clusters Identified:** K-Means successfully grouped the 1,000 users into 4 distinct behavioral clusters.
* **Cluster Interpretation (from `Step3_Clustering_Report.html`):**
    * **Cluster 0 (55%):** "Normal Users" with average, low-risk activity.
    * **Cluster 1 (30%):** "Active Communicators" with high email and external access.
    * **Cluster 2 (12.5%):** "Device-Heavy Users" with high USB connect counts.
    * **Cluster 3 (2.5%):** "Potential Insiders/Outliers." This smallest cluster (25 users) exhibited statistically anomalous behavior, including very high `external_access_count` and a high `is_after_hours_ratio`.
* **Outlier Consensus:** The 25 users in Cluster 3 represent a high-confidence set of anomalies, as their behavior is rare (smallest K-Means cluster) and statistically different (confirmed by Isolation Forest and ANOVA).

---

## 5. File Descriptions

### Python Scripts (`.py`)
* `r4.2_count_rows.py`: Utility to count rows in the original large dataset files.
* `downsampling.py`: **(STEP 1)** Main script to filter users, create/sample sessions, and filter all activity logs based on sampled sessions.
* `downsampling_http_csv.py`: **(STEP 1 - OPTIMIZED)** A batch-processing version of the filtering step specifically for the massive `http.csv` file.
* `feature_engineering.py`: **(STEP 2)** Reads all `filtered_*.csv` files and generates the final `user_features.csv` matrix.
* `elbow_plot_optimal_cluster_no.py`: **(STEP 3.1)** Runs K-Means from k=1-10 to generate data for the elbow plot.
* `k_means_clustering.py`: **(STEP 3.2)** Applies K-Means (k=4) and saves the smallest cluster as `kmeans_outliers.csv`.
* `outlier_analysis.py`: **(STEP 3.3)** Applies Isolation Forest and saves all users with their outlier label (`-1` or `1`).
* `univariate_anova_test.py`: **(STEP 4.1)** Runs K-Means *and* performs ANOVA on all features against the cluster labels, saving results to `univariate_anova_results.csv`.
* `isolation_forest_tsne.py`: **(STEP 4.2)** Generates 2D/3D t-SNE plots colored by Isolation Forest results.
* `k_means_tsne.py`: **(STEP 4.3)** Generates 2D/3D t-SNE plots colored by K-Means outlier status.

### Key Outputs & Reports
* `filtered_output/` (Folder): This directory contains all the intermediate and final data files.
* `filtered_output/filtered_*.csv`: The downsampled data files (ldap, psychometric, logon_sessions, email, file, http, device).
* `filtered_output/user_features.csv`: **(Key File)** The final 1000-row user-feature matrix used for all modeling.
* `filtered_output/kmeans_outliers.csv`: List of users identified as outliers by K-Means (Cluster 3).
* `filtered_output/isolation_forest_outliers.csv`: List of all users and their Isolation Forest outlier status.
* `filtered_output/univariate_anova_results.csv`: Table of all features and their F-value/P-value from the ANOVA test, indicating their significance in separating clusters.
* `*.png`: All plot outputs (Elbow plot, 2D/3D t-SNE plots).
* `Step*.html` / `Final_Report.html`: Detailed HTML reports documenting the methodology, statistical justification, and results for each phase of the project.

---

## 6. How to Run (Execution Order)

To reproduce the analysis, run the scripts in the following order:

1.  **Setup:** Complete all steps in the "Setup & Configuration" section (install dependencies, download data, and **set the `base_data_dir` path in all `.py` scripts**).

2.  **Data Filtering:**
    * Run `downsampling.py` (Note: This will process all files *except* `http.csv` as configured in that script).
    * Run `downsampling_http_csv.py` to process `http.csv` using the memory-safe batch method.
    * *(This generates all `filtered_*.csv` files in the `filtered_output/` folder)*

3.  **Feature Engineering:**
    * Run `feature_engineering.py`.
    * *(This generates `user_features.csv`)*

4.  **Modeling & Analysis:**
    * Run `elbow_plot_optimal_cluster_no.py` (to confirm k=4).
    * Run `k_means_clustering.py` (to generate `kmeans_outliers.csv`).
    * Run `outlier_analysis.py` (to generate `isolation_forest_outliers.csv`).
    * Run `univariate_anova_test.py` (to generate `univariate_anova_results.csv`).

5.  **Visualization:**
    * Run `isolation_forest_tsne.py`.
    * Run `k_means_tsne.py`.

6.  **Review Results:**
    * Open `Final_Report.html` or the individual `Step*.html` files to review the full analysis and interpretation.
