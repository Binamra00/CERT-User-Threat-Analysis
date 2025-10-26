import pandas as pd
import numpy as np
import pathlib
from sklearn.preprocessing import StandardScaler
import math # Import math just in case

# --- Configuration ---
# Use raw string literal r"" or replace backslashes with forward slashes
base_data_dir = pathlib.Path(r"")
output_dir = base_data_dir / "filtered_output"
feature_output_path = output_dir / "user_features.csv"

# --- Load Filtered Files ---
print("Loading filtered datasets...")

# Function to load CSV with error handling
def load_csv_with_error_handling(file_path):
    """Loads a CSV file with error handling for common issues."""
    try:
        # Specify low_memory=False if you encounter DtypeWarning
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded {file_path} ({len(df)} rows)")
        return df
    except PermissionError as e:
        print(f"PermissionError: Unable to read {file_path}. Ensure the file is not open and you have read permissions. Error: {e}")
        raise
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {file_path} does not exist. Please check the file path. Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while loading {file_path}: {e}")
        raise

# LDAP
print("\n--- Processing LDAP ---")
ldap_df = load_csv_with_error_handling(output_dir / "filtered_ldap.csv")
ldap_df = ldap_df.astype({'user_id': 'str'})
# Rename user_id to user for consistency
ldap_df = ldap_df.rename(columns={'user_id': 'user'})
ldap_df['is_manager'] = ldap_df['role'].str.contains('manager', case=False, na=False).astype(int)
ldap_df['is_admin'] = ldap_df['role'].str.contains('admin', case=False, na=False).astype(int)
# Add dummy columns for demonstration (replace with actual columns if available)
if 'department' not in ldap_df.columns:
     ldap_df['department'] = ldap_df['role'].apply(lambda x: 'IT' if any(dept in str(x).upper() for dept in ['IT', 'ENGINEER', 'TECH']) else ('HR' if 'HR' in str(x).upper() else 'Other'))
ldap_df['role'] = ldap_df['role'].str.lower().fillna('unknown')
if 'team_size' not in ldap_df.columns:
    ldap_df['team_size'] = np.random.randint(5, 20, size=len(ldap_df))  # Dummy
if 'tenure' not in ldap_df.columns:
    ldap_df['tenure'] = np.random.randint(1, 10, size=len(ldap_df))  # Dummy
if 'has_external_access' not in ldap_df.columns:
    ldap_df['has_external_access'] = np.random.choice([0, 1], size=len(ldap_df))  # Dummy
if 'security_clearance_level' not in ldap_df.columns:
     ldap_df['security_clearance_level'] = np.random.choice(['low', 'medium', 'high'], size=len(ldap_df)) # Dummy

ldap_features = ldap_df[['user', 'is_manager', 'is_admin', 'department', 'role', 'team_size', 'tenure', 'has_external_access', 'security_clearance_level']]
print(f"LDAP features selected: {list(ldap_features.columns[1:])}")

# Psychometric
print("\n--- Processing Psychometric ---")
psychometric_df = load_csv_with_error_handling(output_dir / "filtered_psychometric.csv")
psychometric_df = psychometric_df.astype({'user_id': 'str'})
# Rename user_id to user for consistency
psychometric_df = psychometric_df.rename(columns={'user_id': 'user'})
# Use actual OCEAN scores from the dataset instead of dummy data
# Ensure OCEAN columns exist before calculating risk score
ocean_cols_check = ['O', 'C', 'E', 'A', 'N']
if all(col in psychometric_df.columns for col in ocean_cols_check):
    psychometric_df['risk_score'] = psychometric_df['N'] - psychometric_df['C']  # Example: Neuroticism - Conscientiousness
    psychometric_features = psychometric_df[['user', 'O', 'C', 'E', 'A', 'N', 'risk_score']]
    # Rename OCEAN columns for clarity
    psychometric_features = psychometric_features.rename(columns={
        'O': 'openness',
        'C': 'conscientiousness',
        'E': 'extraversion',
        'A': 'agreeableness',
        'N': 'neuroticism'
    })
    print(f"Base Psychometric features selected: {list(psychometric_features.columns[1:])}")
else:
    print("Warning: OCEAN columns (O, C, E, A, N) not found in psychometric data. Skipping psychometric features.")
    # Create an empty DataFrame with user column to allow join
    psychometric_features = pd.DataFrame({'user': ldap_features['user'].unique()})


# Logon Sessions
print("\n--- Processing Logon Sessions ---")
logon_df = load_csv_with_error_handling(output_dir / "filtered_logon_sessions.csv")
logon_df = logon_df.astype({'user': 'str'}) # Ensure user is string type
logon_df['logon_time'] = pd.to_datetime(logon_df['logon_time'], errors='coerce')
logon_df['logoff_time'] = pd.to_datetime(logon_df['logoff_time'], errors='coerce')

# Debug: Check for missing times BEFORE dropping
print(f"Initial missing logon_time values: {logon_df['logon_time'].isna().sum()}")
print(f"Initial missing logoff_time values: {logon_df['logoff_time'].isna().sum()}")

# Drop rows where conversion failed or essential time data is missing
logon_df.dropna(subset=['user', 'logon_time', 'logoff_time'], inplace=True)
print(f"Rows after dropping NaNs in time columns: {len(logon_df)}")


if not logon_df.empty:
    logon_df['hour_of_day'] = logon_df['logon_time'].dt.hour
    # Debug: Check hour_of_day values
    print(f"Sample hour_of_day values:\n{logon_df['hour_of_day'].head()}")

    logon_df['session_duration'] = (logon_df['logoff_time'] - logon_df['logon_time']).dt.total_seconds()
    # Filter out negative durations if any (logoff before logon)
    logon_df = logon_df[logon_df['session_duration'] >= 0].copy() # Use .copy() to avoid SettingWithCopyWarning
    print(f"Rows after filtering negative session durations: {len(logon_df)}")

    logon_df['is_after_hours'] = (~logon_df['hour_of_day'].between(9, 17, inclusive='both')).astype(int)
    logon_df['is_weekend'] = logon_df['logon_time'].dt.weekday.isin([5, 6]).astype(int)

    # Compute login hour deviation per user (use transform for direct assignment)
    logon_df['hour_mean'] = logon_df.groupby('user')['hour_of_day'].transform('mean')
    logon_df['login_hour_deviation'] = (logon_df['hour_of_day'] - logon_df['hour_mean']).abs()

    # Aggregate at user level
    logon_features = logon_df.groupby('user').agg(
        hour_of_day_mean=('hour_of_day', 'mean'),
        hour_of_day_variance=('hour_of_day', 'var'),
        login_count_agg=('user', 'count'), # Temp name to avoid potential conflict if 'count' exists
        session_duration_mean=('session_duration', 'mean'),
        session_duration_max=('session_duration', 'max'),
        session_duration_min=('session_duration', 'min'),
        session_duration_variance=('session_duration', 'var'),
        login_hour_deviation_mean=('login_hour_deviation', 'mean'),
        is_after_hours_ratio=('is_after_hours', 'mean'),
        is_weekend_ratio=('is_weekend', 'mean') # Renamed for clarity
    ).reset_index()

    # Rename columns after aggregation
    logon_features = logon_features.rename(columns={
        'login_count_agg': 'login_count',
        'is_weekend_ratio': 'weekend_login_ratio' # Match previous naming if desired
    })

    # Calculate session frequency
    total_days = (logon_df['logon_time'].max() - logon_df['logon_time'].min()).days + 1
    total_days = max(1, total_days) # Ensure at least 1 day
    logon_features['session_frequency'] = logon_features['login_count'] / total_days

    # Ensure all 11 features are present
    logon_features = logon_features[['user', 'hour_of_day_mean', 'hour_of_day_variance', 'login_count',
                                     'session_duration_mean', 'session_duration_max', 'session_duration_min',
                                     'session_duration_variance', 'login_hour_deviation_mean',
                                     'is_after_hours_ratio', 'weekend_login_ratio', 'session_frequency']]

    print(f"Logon features selected ({len(logon_features.columns[1:])}): {list(logon_features.columns[1:])}")
else:
    print("Warning: Logon DataFrame is empty after cleaning. Skipping logon feature generation.")
    logon_features = pd.DataFrame({'user': ldap_features['user'].unique()})


# Email
print("\n--- Processing Email ---")
email_df = load_csv_with_error_handling(output_dir / "filtered_email.csv")
email_df = email_df.astype({'user': 'str'})
email_df['date'] = pd.to_datetime(email_df['date'], errors='coerce')
email_df.dropna(subset=['user', 'date'], inplace=True) # Drop essential NaNs

if not email_df.empty:
    email_df['hour'] = email_df['date'].dt.hour
    email_df['is_after_hours'] = (~email_df['hour'].between(9, 17, inclusive='both')).astype(int)
    # Debug: Inspect size and to fields
    if 'size' in email_df.columns:
        print(f"Email size range: min={email_df['size'].min()}, max={email_df['size'].max()}")
        # Adjust threshold for large attachments based on actual size range (e.g., top 10%)
        size_threshold = email_df['size'].quantile(0.9)
        email_df['has_large_attachment'] = (email_df['size'] > size_threshold).astype(int)
        email_df['attachment_size'] = email_df['size'] / 1_000_000  # Convert bytes to MB (or keep as bytes)
    else:
        print("Warning: 'size' column not found in email_df. Setting attachment features to 0.")
        email_df['has_large_attachment'] = 0
        email_df['attachment_size'] = 0

    if 'to' in email_df.columns:
        print(f"Sample to field values:\n{email_df['to'].head()}")
        email_df['to'] = email_df['to'].fillna('') # Handle NaN in 'to' field
        # Update external email detection - ADJUST 'company.com' domain
        email_df['is_external'] = (~email_df['to'].str.contains('@dtaa.com', case=False, na=False)).astype(int)
        # Compute unique recipients per email (handle potential errors)
        try:
            email_df['unique_recipients'] = email_df['to'].str.split(';').apply(lambda x: len(set(filter(None, x))) if isinstance(x, list) else 0)
        except Exception as e:
            print(f"Error processing 'to' field for unique_recipients: {e}. Setting to 0.")
            email_df['unique_recipients'] = 0
        print(f"Sample unique_recipients values:\n{email_df['unique_recipients'].head()}")
        # Calculate unique domains per email (can be slow, consider approximation or sampling if needed)
        try:
             email_df['unique_domains_count'] = email_df['to'].str.findall(r'@([\w\.-]+)').apply(lambda x: len(set(y.lower() for y in x)) if isinstance(x, list) else 0)
        except Exception as e:
            print(f"Error processing 'to' field for unique_domains: {e}. Setting to 0.")
            email_df['unique_domains_count'] = 0
    else:
        print("Warning: 'to' column not found in email_df. Setting related features (external, recipients, domains) to 0.")
        email_df['is_external'] = 0
        email_df['unique_recipients'] = 0
        email_df['unique_domains_count'] = 0


    email_df['is_weekend'] = email_df['date'].dt.weekday.isin([5, 6]).astype(int)

    # Aggregate at user level
    email_features = email_df.groupby('user').agg(
        unique_recipients_mean=('unique_recipients', 'mean'),
        has_large_attachment_ratio=('has_large_attachment', 'mean'),
        external_email_ratio=('is_external', 'mean'),
        external_email_count=('is_external', 'sum'),
        email_hour_mean=('hour', 'mean'),
        email_hour_variance=('hour', 'var'),
        after_hours_email_ratio=('is_after_hours', 'mean'),
        email_unique_domains_mean=('unique_domains_count', 'mean'), # Aggregate the count calculated per email
        attachment_size_mean=('attachment_size', 'mean'),
        is_weekend_sum=('is_weekend', 'sum'), # Temp for weekday ratio
        email_count_agg=('user', 'count') # Temp name
    ).reset_index()

    email_features = email_features.rename(columns={'email_count_agg': 'email_count'})

    # Calculate derived features
    email_features['weekday_email_ratio'] = 1 - (email_features['is_weekend_sum'] / email_features['email_count'].replace(0, 1))
    # Use total_days calculated earlier if logon_df was processed successfully
    if 'total_days' in locals() and total_days > 0 : # Check if total_days is defined and > 0
         email_features['email_frequency'] = email_features['email_count'] / total_days
    else:
         email_features['email_frequency'] = 0 # Fallback if total_days isn't available or is 0
    email_features['email_frequency_variance'] = email_df.groupby('user')['unique_recipients'].var().reindex(email_features['user'], fill_value=0)

    # Select final 13 features, dropping intermediate is_weekend_sum
    email_features = email_features[['user', 'unique_recipients_mean', 'has_large_attachment_ratio', 'external_email_ratio',
                                     'external_email_count', 'email_hour_mean', 'email_hour_variance',
                                     'after_hours_email_ratio', 'email_unique_domains_mean', 'attachment_size_mean',
                                     'weekday_email_ratio', 'email_count', 'email_frequency', 'email_frequency_variance']]

    print(f"Email features selected ({len(email_features.columns[1:])}): {list(email_features.columns[1:])}")
else:
    print("Warning: Email DataFrame is empty after cleaning. Skipping email feature generation.")
    email_features = pd.DataFrame({'user': ldap_features['user'].unique()})


# Device
print("\n--- Processing Device ---")
device_df = load_csv_with_error_handling(output_dir / "filtered_device.csv")
device_df = device_df.astype({'user': 'str'})
device_df['date'] = pd.to_datetime(device_df['date'], errors='coerce')
device_df.dropna(subset=['user', 'date'], inplace=True)

if not device_df.empty:
    device_df['hour'] = device_df['date'].dt.hour
    device_df['is_after_hours'] = (~device_df['hour'].between(9, 17, inclusive='both')).astype(int)

    # Check columns before calculating features
    if 'activity' in device_df.columns:
        device_df['is_connect'] = (device_df['activity'] == 'Connect').astype(int)
        device_df['is_disconnect'] = (device_df['activity'] == 'Disconnect').astype(int)
        print(f"Unique device activity values: {device_df['activity'].unique()}")
        # Update USB logic based on actual activity values if possible
        # Example: If USB connection has a specific activity string
        # device_df['is_usb'] = device_df['activity'].str.contains('USB', case=False, na=False).astype(int)
        # Simple assumption from your code: Connect events are USB-related for counting
        device_df['is_usb'] = (device_df['activity'] == 'Connect').astype(int)
    else:
        print("Warning: 'activity' column missing in device_df. Setting connect/disconnect/usb features to 0.")
        device_df['is_connect'] = 0
        device_df['is_disconnect'] = 0
        device_df['is_usb'] = 0

    # Check for 'pc' column
    pc_col_exists = 'pc' in device_df.columns
    if not pc_col_exists:
        print("Warning: 'pc' column missing in device_df. Cannot calculate unique_pcs.")
        # Add a dummy 'pc' column if needed for aggregation, or remove unique_pcs from agg
        # device_df['pc'] = 'unknown_pc' # Option: Add dummy


    # Aggregate at user level
    agg_dict_device = {
        'connect_count': ('is_connect', 'sum'), # Keep intermediate count
        'disconnect_count': ('is_disconnect', 'sum'), # Keep intermediate count
        'device_hour_mean': ('hour', 'mean'),
        'device_hour_variance': ('hour', 'var'),
        'after_hours_device_ratio': ('is_after_hours', 'mean'),
        'usb_connect_count': ('is_usb', 'sum'),
        'device_event_count_agg': ('user', 'count') # Temp name
    }
    if pc_col_exists:
        agg_dict_device['unique_pcs'] = ('pc', 'nunique')

    device_features = device_df.groupby('user').agg(**agg_dict_device).reset_index()


    device_features = device_features.rename(columns={'device_event_count_agg': 'device_event_count'})

    # Calculate derived features
    device_features['connect_disconnect_ratio'] = device_features['connect_count'] / device_features['disconnect_count'].replace(0, 1)
    if 'total_days' in locals() and total_days > 0:
        device_features['device_usage_frequency'] = device_features['device_event_count'] / total_days
    else:
        device_features['device_usage_frequency'] = 0

    # Select final features, dropping intermediate counts
    final_device_cols = ['user', 'connect_disconnect_ratio', 'device_hour_mean',
                         'device_hour_variance', 'after_hours_device_ratio', 'usb_connect_count',
                         'device_event_count', 'device_usage_frequency']
    if pc_col_exists:
         final_device_cols.insert(2, 'unique_pcs') # Insert 'unique_pcs' if calculated

    device_features = device_features[final_device_cols]

    print(f"Device features selected ({len(device_features.columns[1:])}): {list(device_features.columns[1:])}")
else:
    print("Warning: Device DataFrame is empty after cleaning. Skipping device feature generation.")
    device_features = pd.DataFrame({'user': ldap_features['user'].unique()})


# File
print("\n--- Processing File ---")
file_df = load_csv_with_error_handling(output_dir / "filtered_file.csv")
file_df = file_df.astype({'user': 'str'})
file_df['date'] = pd.to_datetime(file_df['date'], errors='coerce') # Convert date column first
file_df.dropna(subset=['user', 'date'], inplace=True) # Drop essential NaNs

if not file_df.empty:
    # Create the 'hour' column by extracting from the 'date' column
    file_df['hour'] = file_df['date'].dt.hour

    # Now you can use the 'hour' column
    file_df['is_after_hours'] = (~file_df['hour'].between(9, 17, inclusive='both')).astype(int)

    # Debug: Inspect filename and content
    if 'filename' in file_df.columns:
        print(f"Sample filename values:\n{file_df['filename'].head()}")
        file_df['filename'] = file_df['filename'].fillna('') # Handle NaN
        file_df['file_type'] = file_df['filename'].str.extract(r'\.(\w+)$')[0].fillna('unknown')
    else:
        print("Warning: 'filename' column missing in file_df. Setting file_type to 'unknown'.")
        file_df['filename'] = ''
        file_df['file_type'] = 'unknown'

    if 'content' in file_df.columns:
        print(f"Sample content values:\n{file_df['content'].head()}")
        # Adjust sensitive keywords as needed
        file_df['is_sensitive'] = file_df['content'].str.contains('proprietary|sensitive|restricted|confidential|secret', case=False, na=False).astype(int)
    else:
        print("Warning: 'content' column not found in file_df. Setting 'is_sensitive' to 0.")
        file_df['is_sensitive'] = 0

    # Placeholder for external drive - requires specific logic based on data (e.g., path patterns)
    # Example: Check if path starts with typical removable drive letters
    # file_df['is_external_drive'] = file_df['filename'].str.contains(r'^[d-zD-Z]:\\', case=False, na=False, regex=True).astype(int)
    file_df['is_external_drive'] = 0 # Keep placeholder for now

    file_features = file_df.groupby('user').agg(
        unique_file_types=('file_type', 'nunique'),
        sensitive_file_access_count=('is_sensitive', 'sum'),
        sensitive_file_ratio_agg=('is_sensitive', 'mean'), # Rename agg result
        file_hour_mean=('hour', 'mean'),
        file_hour_variance=('hour', 'var'),
        after_hours_file_ratio=('is_after_hours', 'mean'),
        unique_filenames_mean=('filename', 'nunique'), # Using nunique as proxy for mean unique per session
        external_drive_access_count=('is_external_drive', 'sum'),
        file_access_count_agg=('user', 'count') # Temp name
    ).reset_index()

    file_features = file_features.rename(columns={
        'sensitive_file_ratio_agg': 'sensitive_file_ratio',
        'file_access_count_agg': 'file_access_count'
        })

    if 'total_days' in locals() and total_days > 0:
        file_features['file_access_frequency'] = file_features['file_access_count'] / total_days
    else:
         file_features['file_access_frequency'] = 0

    # Calculate Shannon entropy for file type diversity
    file_features['file_type_diversity'] = file_df.groupby('user')['file_type'].apply(lambda x: -sum(p * np.log2(p) for p in x.value_counts(normalize=True) if p > 0)).reindex(file_features['user'], fill_value=0)

    # Select final features
    file_features = file_features[['user', 'unique_file_types', 'sensitive_file_access_count', 'sensitive_file_ratio',
                                   'file_hour_mean', 'file_hour_variance', 'after_hours_file_ratio',
                                   'unique_filenames_mean', 'external_drive_access_count', 'file_access_count',
                                   'file_access_frequency', 'file_type_diversity']]

    print(f"File features selected ({len(file_features.columns[1:])}): {list(file_features.columns[1:])}")
else:
    print("Warning: File DataFrame is empty after cleaning. Skipping file feature generation.")
    file_features = pd.DataFrame({'user': ldap_features['user'].unique()})


# HTTP
print("\n--- Processing HTTP ---")
http_df = load_csv_with_error_handling(output_dir / "filtered_http.csv")
http_df = http_df.astype({'user': 'str'})
http_df['date'] = pd.to_datetime(http_df['date'], errors='coerce')
http_df.dropna(subset=['user', 'date'], inplace=True) # Drop essential NaNs

if not http_df.empty:
    http_df['hour'] = http_df['date'].dt.hour
    # Debug: Check date parsing and URLs
    print(f"Missing HTTP date values after conversion: {http_df['date'].isna().sum()}")

    if 'url' in http_df.columns:
         print(f"Sample URL values:\n{http_df['url'].head()}")
         http_df['url'] = http_df['url'].fillna('') # Handle NaN
         # Adjust company domain and risky keywords as needed
         http_df['is_external'] = (~http_df['url'].str.contains('dtaa.com', case=False, na=False)).astype(int)
         http_df['url_length'] = http_df['url'].str.len().fillna(0)
         http_df['is_risky'] = http_df['url'].str.contains('dropbox|pastebin|darkweb|mega.nz|mediafire', case=False, na=False).astype(int)
    else:
        print("Warning: 'url' column missing in http_df. Setting related features to 0/empty.")
        http_df['url'] = ''
        http_df['is_external'] = 0
        http_df['url_length'] = 0
        http_df['is_risky'] = 0

    http_df['is_after_hours'] = (~http_df['hour'].between(9, 17, inclusive='both')).astype(int)
    http_df['is_weekend'] = http_df['date'].dt.weekday.isin([5, 6]).astype(int)

    if 'content' in http_df.columns:
        http_df['content'] = http_df['content'].fillna('') # Handle NaN
        http_df['content_keyword'] = http_df['content'].str.contains('confidential|secret|password|proprietary', case=False, na=False).astype(int)
    else:
        print("Warning: 'content' column missing in http_df. Setting 'content_keyword' to 0.")
        http_df['content_keyword'] = 0

    # Aggregate without including 'user' in the agg dictionary to avoid MultiIndex issues
    http_features = http_df.groupby('user').agg(
        external_access_count=('is_external', 'sum'),
        external_access_ratio=('is_external', 'mean'),
        http_hour_mean=('hour', 'mean'),
        http_hour_variance=('hour', 'var'),
        after_hours_http_ratio=('is_after_hours', 'mean'),
        url_nunique=('url', 'nunique'), # Use direct aggregation for nunique
        is_weekend_mean=('is_weekend', 'mean'), # For weekday ratio
        url_length_mean=('url_length', 'mean'),
        risky_domain_count=('is_risky', 'sum'),
        content_keyword_count=('content_keyword', 'sum'),
        http_access_count_agg = ('user', 'count') # Get count directly
    ).reset_index()

    # Rename columns for clarity and consistency
    http_features = http_features.rename(columns={
        'url_nunique': 'unique_urls_mean', # Match report naming intent
        'is_weekend_mean': 'weekend_http_ratio', # Keep consistent naming? Or calculate weekday?
        'http_access_count_agg': 'http_access_count'
    })
    # Calculate weekday ratio if needed: 1 - weekend_ratio
    http_features['weekday_http_ratio'] = 1 - http_features['weekend_http_ratio']


    # Add additional features calculated separately
    if 'total_days' in locals() and total_days > 0:
        http_features['http_access_frequency_mean'] = http_features['http_access_count'] / total_days
    else:
        http_features['http_access_frequency_mean'] = 0
    http_features['http_frequency_variance'] = http_df.groupby('user')['hour'].var().reindex(http_features['user'], fill_value=0)

    # Calculate unique domains mean carefully
    try:
        # Extract domain: handles http/https, optional www., stops at first '/'
        http_df['domain'] = http_df['url'].str.extract(r'https?://(?:www\.)?([\w\.-]+)', expand=False).fillna('unknown')
        http_unique_domains_calc = http_df.groupby('user')['domain'].nunique().reset_index()
        http_unique_domains_calc.columns = ['user', 'http_unique_domains_mean']
        http_features = pd.merge(http_features, http_unique_domains_calc, on='user', how='left')
        http_features['http_unique_domains_mean'] = http_features['http_unique_domains_mean'].fillna(0) # Fill NaNs if merge fails for some users
    except Exception as e:
        print(f"Error calculating http_unique_domains_mean: {e}. Setting to 0.")
        http_features['http_unique_domains_mean'] = 0


    # Select final features (ensure correct columns are selected)
    final_http_cols = ['user', 'external_access_count', 'external_access_ratio', 'http_hour_mean',
                       'http_hour_variance', 'after_hours_http_ratio', 'unique_urls_mean',
                       'weekday_http_ratio', 'url_length_mean', 'risky_domain_count',
                       'content_keyword_count', 'http_access_count', 'http_access_frequency_mean',
                       'http_frequency_variance', 'http_unique_domains_mean']
    # Ensure all columns exist before selecting
    final_http_cols = [col for col in final_http_cols if col in http_features.columns]
    http_features = http_features[final_http_cols]


    print(f"HTTP features selected ({len(http_features.columns[1:])}): {list(http_features.columns[1:])}")
else:
    print("Warning: HTTP DataFrame is empty after cleaning. Skipping HTTP feature generation.")
    http_features = pd.DataFrame({'user': ldap_features['user'].unique()})


# --- Combine Features ---
print("\nCombining features into user-level dataset...")

# Ensure ldap_features has 'user' as index before starting join
if 'user' in ldap_features.columns:
    user_features = ldap_features.set_index('user')
else:
    print("Error: 'user' column not found in ldap_features. Cannot start join.")
    # Handle error: Maybe exit or create an empty user_features DataFrame
    user_features = pd.DataFrame() # Or sys.exit()


if not user_features.empty:
    # Dictionary of feature DataFrames to join
    feature_dfs = {
        'psychometric': psychometric_features,
        'logon': logon_features,
        'email': email_features,
        'device': device_features,
        'file': file_features,
        'http': http_features
    }

    for name, df in feature_dfs.items():
        print(f"Attempting to join {name}...")
        # Ensure df is a DataFrame and not empty before proceeding
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"Skipping join for non-DataFrame or empty DataFrame: {name}")
            continue
        if 'user' not in df.columns:
            print(f"Error: 'user' column missing in {name}. Skipping join.")
            continue

        # Check for duplicate columns before join (excluding the 'user' index)
        df_to_join = df.set_index('user') # Set index for joining
        overlapping_cols = user_features.columns.intersection(df_to_join.columns)
        if len(overlapping_cols) > 0:
            print(f"Warning: Overlapping columns found joining {name}: {list(overlapping_cols)}. Columns from {name} will be ignored.")
            # Drop overlapping columns from the right DataFrame before joining
            df_to_join = df_to_join.drop(columns=overlapping_cols)

        if not df_to_join.empty:
             # Ensure the index of user_features is string type for consistent joins
             user_features.index = user_features.index.astype(str)
             df_to_join.index = df_to_join.index.astype(str)
             user_features = user_features.join(df_to_join, how='left')
             print(f"Successfully joined {name}. Current shape: {user_features.shape}")
        else:
            print(f"Skipping join for {name} as no new columns remained after handling overlaps or it was empty.")


    # Add interaction features if base columns exist
    user_features = user_features.reset_index() # Get 'user' back as column

    required_for_interaction = ['neuroticism', 'external_access_count', 'conscientiousness', 'is_after_hours_ratio']
    missing_cols = [col for col in required_for_interaction if col not in user_features.columns]

    if not missing_cols:
        print("Adding interaction features...")
        # Fill NaNs that might result from left joins before calculating interactions
        # Use median for psych scores, 0 for counts/ratios makes sense here
        user_features['neuroticism'] = user_features['neuroticism'].fillna(user_features['neuroticism'].median())
        user_features['external_access_count'] = user_features['external_access_count'].fillna(0)
        user_features['conscientiousness'] = user_features['conscientiousness'].fillna(user_features['conscientiousness'].median())
        user_features['is_after_hours_ratio'] = user_features['is_after_hours_ratio'].fillna(0)

        user_features['neuroticism_external_access_interaction'] = user_features['neuroticism'] * user_features['external_access_count']
        user_features['conscientiousness_after_hours_interaction'] = user_features['conscientiousness'] * user_features['is_after_hours_ratio']
    else:
        print(f"Warning: Cannot create interaction features. Base columns missing: {missing_cols}")
        user_features['neuroticism_external_access_interaction'] = 0 # Add columns as 0 if base missing
        user_features['conscientiousness_after_hours_interaction'] = 0


    # --- Clean and Normalize ---
    print("Cleaning and Normalizing...")
    # Fill NaNs resulting from left joins or calculations
    # Strategy: Fill numerical features potentially with 0 or median/mean
    # Categorical features (like department, role, security_clearance_level) with 'Missing' or mode
    numerical_cols_final = user_features.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_final = user_features.select_dtypes(exclude=np.number).columns.tolist()

    if 'user' in numerical_cols_final: numerical_cols_final.remove('user')
    if 'user' in categorical_cols_final: categorical_cols_final.remove('user')

    # Fill numerical NaNs (e.g., with median) - Calculate medians first
    medians = user_features[numerical_cols_final].median()
    user_features[numerical_cols_final] = user_features[numerical_cols_final].fillna(medians)
    # Check if any NaNs remain in numerical (median could be NaN if column was all NaN)
    remaining_num_nans = user_features[numerical_cols_final].isna().sum()
    cols_still_nan = remaining_num_nans[remaining_num_nans > 0].index.tolist()
    if cols_still_nan:
        print(f"Warning: Columns still contain NaNs after median fill (likely all-NaN originally): {cols_still_nan}. Filling these with 0.")
        user_features[cols_still_nan] = user_features[cols_still_nan].fillna(0)


    # Fill categorical NaNs (e.g., with 'Missing')
    for col in categorical_cols_final:
        user_features[col] = user_features[col].fillna('Missing')

    print("NaNs filled (Numerical with Median then 0, Categorical with 'Missing').")


    # One-Hot Encode categorical features before scaling
    if categorical_cols_final:
        print(f"One-Hot Encoding categorical columns: {categorical_cols_final}")
        user_features = pd.get_dummies(user_features, columns=categorical_cols_final, dummy_na=False, drop_first=False) # drop_first=True can reduce multicollinearity
        print(f"Shape after One-Hot Encoding: {user_features.shape}")
        # Update numerical cols list to include new OHE columns and exclude original categoricals
        numerical_cols_final = user_features.select_dtypes(include=np.number).columns.tolist()
        if 'user' in numerical_cols_final: numerical_cols_final.remove('user')


    # --- ADD THIS BLOCK TO REMOVE ZERO-VARIANCE COLUMNS ---
    print(f"\nInitial number of numerical columns for scaling: {len(numerical_cols_final)}")
    # Calculate variance for numerical columns intended for scaling
    # Ensure columns exist before calculating variance
    cols_to_check_variance = [col for col in numerical_cols_final if col in user_features.columns]
    if cols_to_check_variance:
        variances = user_features[cols_to_check_variance].var()
        # Identify columns with zero (or near-zero) variance
        zero_variance_cols = variances[variances < 1e-8].index.tolist() # Using threshold for float comparison

        if zero_variance_cols:
            print(f"Warning: Found {len(zero_variance_cols)} columns with zero variance:")
            # Print first few examples if list is long
            print(zero_variance_cols[:10] if len(zero_variance_cols) > 10 else zero_variance_cols)
            print("Removing these columns before scaling.")
            user_features = user_features.drop(columns=zero_variance_cols)
            # Update the list of numerical columns to scale
            numerical_cols_final = [col for col in numerical_cols_final if col not in zero_variance_cols]
            print(f"Number of numerical columns remaining for scaling: {len(numerical_cols_final)}")
        else:
            print("No zero-variance columns found.")
    else:
        print("No numerical columns found to check variance.")
    # --- END OF BLOCK ---


    # Normalize numerical features (Now safe from zero-variance columns)
    print(f"\nNormalizing {len(numerical_cols_final)} numerical features...")
    if numerical_cols_final:
        scaler = StandardScaler()
        # Ensure the columns still exist after potential removal
        valid_numerical_cols = [col for col in numerical_cols_final if col in user_features.columns]
        if valid_numerical_cols:
            user_features[valid_numerical_cols] = scaler.fit_transform(user_features[valid_numerical_cols])
            print("Numerical features normalized using StandardScaler.")
        else:
            print("No valid numerical columns remained for scaling.")
    else:
        print("No numerical columns identified for scaling.")

    # --- Save Features ---
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    user_features.to_csv(feature_output_path, index=False)
    final_feature_count = len(user_features.columns) - 1 # Subtract 'user'
    print(f"\nUser-level features dataset saved to {feature_output_path}")
    print(f"Final shape: {user_features.shape}")
    print(f"Total features extracted (after OHE, excluding user ID): {final_feature_count}")

else:
    print("Processing stopped as initial LDAP feature DataFrame was empty.")