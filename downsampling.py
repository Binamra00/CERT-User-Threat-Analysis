import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pathlib
import traceback

# --- Configuration ---
base_data_dir = pathlib.Path(r"")
ldap_dir = base_data_dir / "LDAP"
output_dir = base_data_dir / "filtered_output"

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---
def create_sessions_for_user_partition(user_df):
    user_df = user_df.sort_values(['user', 'date'])
    sessions = []

    for user, group in user_df.groupby('user'):
        logons = group[group['activity'] == 'Logon'].copy()
        logoffs = group[group['activity'] == 'Logoff'].copy()

        logons['next_logoff_time'] = logons['date'].apply(
            lambda logon_time: logoffs[logoffs['date'] > logon_time]['date'].min()
        )

        logons['next_logoff_time'] = logons['next_logoff_time'].fillna(
            logons['date'].apply(lambda dt: dt.replace(hour=23, minute=59, second=59))
        )

        for _, logon in logons.iterrows():
            logon_time = logon['date']
            logoff_time = logon['next_logoff_time']
            duration_hours = (logoff_time - logon_time).total_seconds() / 3600

            if duration_hours > 0:
                sessions.append({
                    'user': user,
                    'logon_time': logon_time,
                    'logoff_time': logoff_time,
                    'session_duration': duration_hours
                })

    return pd.DataFrame(sessions)

def stratify_and_sample_sessions(user_sessions_df, n_target, time_bins=6, min_pct=0.1, min_abs=10):
    total_sessions = len(user_sessions_df)
    n_required = max(int(min_pct * total_sessions), min_abs)
    n_adjusted = min(n_target, n_required)

    if total_sessions <= n_adjusted:
        return user_sessions_df.copy()

    user_sessions_df = user_sessions_df.copy()
    try:
        user_sessions_df['logon_time'] = pd.to_datetime(user_sessions_df['logon_time'])
        user_sessions_df['time_bin'] = pd.cut(
            user_sessions_df['logon_time'].astype(np.int64) // 10**9,
            bins=time_bins,
            labels=False,
            right=False
        )
        user_sessions_df = user_sessions_df[np.isfinite(user_sessions_df['time_bin'])].copy()
        if user_sessions_df.empty:
            raise ValueError("All rows filtered out after time binning non-finite check.")
    except (ValueError, TypeError) as e:
        user = user_sessions_df['user'].iloc[0] if not user_sessions_df.empty else "Unknown"
        print(f"Warning: Could not create time bins for user {user}. Error: {e}. Falling back to random sampling.")
        return user_sessions_df.sample(n=min(total_sessions, n_adjusted), random_state=42).copy()

    if user_sessions_df['time_bin'].nunique() == 0:
        print(f"Warning: No valid time bins for user {user_sessions_df['user'].iloc[0]}. Falling back to random sampling.")
        return user_sessions_df.sample(n=min(total_sessions, n_adjusted), random_state=42).copy()

    sampled_df_list = []
    pandas_major_minor = tuple(map(int, pd.__version__.split('.')[:2]))
    use_observed = pandas_major_minor >= (1, 1)
    groupby_args = {'group_keys': False}
    if use_observed:
        groupby_args['observed'] = True

    try:
        for bin_id, bin_group in user_sessions_df.groupby('time_bin', **groupby_args):
            if pd.isna(bin_id):
                continue
            bin_size = 1  # 1 session per bin as specified (n_target=6, time_bins=6)
            if not bin_group.empty:
                sampled_df_list.append(bin_group.sample(min(len(bin_group), bin_size), random_state=42))
    except TypeError as e:
        print(f"Warning: Groupby arguments issue ({e}). Falling back to simpler groupby iteration.")
        for bin_id, bin_group in user_sessions_df.groupby('time_bin'):
            if pd.isna(bin_id):
                continue
            bin_size = 1
            if not bin_group.empty:
                sampled_df_list.append(bin_group.sample(min(len(bin_group), bin_size), random_state=42))

    if sampled_df_list:
        sampled_df = pd.concat(sampled_df_list, ignore_index=True)
    else:
        sampled_df = pd.DataFrame(columns=user_sessions_df.columns)

    if len(sampled_df) < n_adjusted:
        needed = n_adjusted - len(sampled_df)
        remaining_df = user_sessions_df.drop(columns=['time_bin'])
        if not sampled_df.empty:
            remaining_df = remaining_df.loc[~remaining_df.index.isin(sampled_df.index)]
        if len(remaining_df) >= needed:
            remaining_df = remaining_df.reset_index(drop=True)
            sampled_df = pd.concat([sampled_df, remaining_df.sample(n=needed, random_state=42)])
        else:
            sampled_df = pd.concat([sampled_df, remaining_df])

    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df.drop(columns=['time_bin']).copy()

# --- Step 1.1: LDAP Preprocessing ---
print("Step 1.1: LDAP Preprocessing")
ldap_files = sorted(list(ldap_dir.glob('*.csv')))
if not ldap_files:
    print(f"Error: No CSV files found in {ldap_dir}")
    exit()

ldap_dfs = []
for file in ldap_files:
    try:
        df = pd.read_csv(file, dtype={'user_id': 'str'})
        file_date_str = file.stem
        df['file_date'] = pd.to_datetime(file_date_str + '-01', format='%Y-%m-%d', errors='coerce')
        ldap_dfs.append(df)
    except Exception as e:
        print(f"Warning: Could not read or process LDAP file {file}: {e}")
        traceback.print_exc()

if not ldap_dfs:
    print("Error: No LDAP data loaded.")
    exit()

ldap_combined = pd.concat(ldap_dfs, ignore_index=True)
ldap_combined = ldap_combined.dropna(subset=['user_id', 'file_date'])
ldap_deduped = ldap_combined.sort_values(by=['user_id', 'file_date'], ascending=[True, False])
ldap_deduped = ldap_deduped.drop_duplicates(subset=['user_id'], keep='first')

selected_users = set(ldap_deduped['user_id'])
selected_users_list = list(selected_users)
ldap_output_file = output_dir / "filtered_ldap.csv"
try:
    ldap_deduped.to_csv(ldap_output_file, index=False)
    print(f"Filtered LDAP users: {len(selected_users)}")
    print(f"File saved: {ldap_output_file}")
except Exception as e:
    print(f"Error saving LDAP file {ldap_output_file}: {e}")
    traceback.print_exc()

# --- Step 1.2: Psychometric Data Filtering ---
print("\nStep 1.2: Psychometric Data Filtering")
psychometric_file = base_data_dir / "psychometric.csv"
psychometric_output_file = output_dir / "filtered_psychometric.csv"
psychometric_df_filtered = pd.DataFrame()

if psychometric_file.exists():
    try:
        psychometric_df = pd.read_csv(psychometric_file, dtype={'user_id': 'str'})
        if 'user_id' in psychometric_df.columns:
            psychometric_df_filtered = psychometric_df[psychometric_df['user_id'].isin(selected_users)].copy()
            psychometric_df_filtered.to_csv(psychometric_output_file, index=False)
            print(f"Filtered psychometric rows: {len(psychometric_df_filtered)}")
            print(f"File saved: {psychometric_output_file}")
        else:
            print(f"Warning: 'user_id' column not found in {psychometric_file}. Skipping filtering.")
    except Exception as e:
        print(f"Warning: Could not read or process {psychometric_file}: {e}")
        traceback.print_exc()
else:
    print(f"File {psychometric_file} not found. Skipping...")

# --- Step 1.3: Session Creation from Logon Data ---
print("\nStep 1.3: Session Creation from Logon Data")
logon_file = base_data_dir / "logon.csv"
if not logon_file.exists():
    print(f"File {logon_file} not found. Exiting...")
    exit()

logon_dtypes = {'id': 'str', 'date': 'str', 'user': 'str', 'pc': 'str', 'activity': 'str'}
try:
    ddf_logon = dd.read_csv(logon_file, dtype=logon_dtypes, blocksize='128MB', assume_missing=True)
    print(f"Original logon rows (estimated): {len(ddf_logon)}")

    ddf_logon = ddf_logon[ddf_logon['user'].isin(selected_users_list)]
    ddf_logon['date'] = dd.to_datetime(ddf_logon['date'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
    ddf_logon = ddf_logon.dropna(subset=['date', 'user'])
    print(f"Logon rows after user filter & date parsing (estimated): {len(ddf_logon)}")

    meta = pd.DataFrame({
        'user': pd.Series([], dtype='object'),
        'logon_time': pd.Series([], dtype='datetime64[ns]'),
        'logoff_time': pd.Series([], dtype='datetime64[ns]'),
        'session_duration': pd.Series([], dtype='float64')
    })

    ddf_sessions = ddf_logon.map_partitions(create_sessions_for_user_partition, meta=meta)
    print("Computing all valid sessions...")
    all_sessions_df = ddf_sessions.compute()
    print(f"Total valid sessions created: {len(all_sessions_df)}")
except Exception as e:
    print(f"Error during Logon processing or session creation: {e}")
    traceback.print_exc()
    all_sessions_df = pd.DataFrame(columns=['user', 'logon_time', 'logoff_time', 'session_duration'])

# --- Step 1.4: Set Fixed Session Sample Size ---
print("\nStep 1.4: Set Fixed Session Sample Size")
n_sessions_target = 6
print(f"Target session sample size per user (fixed): {n_sessions_target}")
try:
    z_score = 1.96
    p = 0.5
    if n_sessions_target > 0:
        estimated_e = np.sqrt((z_score**2 * p * (1-p)) / n_sessions_target)
        print(f"Note: This fixed sample size ({n_sessions_target}) corresponds to an estimated margin of error of ±{estimated_e:.3f} ({estimated_e*100:.1f}%) at 95% confidence.")
    else:
        print("Note: Target sample size is 0, no sessions will be sampled.")
except Exception as e:
    print(f"Could not estimate margin of error for fixed sample size: {e}")

# --- Step 1.5: Session Downsampling with Temporal Stratification ---
print("\nStep 1.5: Session Downsampling")
sampled_sessions_df = pd.DataFrame(columns=['user', 'logon_time', 'logoff_time', 'session_duration'])

if not all_sessions_df.empty and n_sessions_target > 0:
    print("Applying stratified sampling per user...")
    sampled_sessions_list = []
    try:
        if 'user' in all_sessions_df.columns:
            for user, group in all_sessions_df.groupby('user'):
                if not group.empty:
                    sampled_group = stratify_and_sample_sessions(group, n_target=n_sessions_target)
                    if not sampled_group.empty:
                        sampled_sessions_list.append(sampled_group)
            if sampled_sessions_list:
                sampled_sessions_df = pd.concat(sampled_sessions_list, ignore_index=True)
                sampled_sessions_df = sampled_sessions_df[sampled_sessions_df.columns.intersection(sampled_sessions_df.columns)].copy()
                sampled_sessions_df = sampled_sessions_df.sort_values(['user', 'logon_time'])
            else:
                print("Warning: No sessions sampled for any user.")
        else:
            print("Error: 'user' column not found in all_sessions_df. Skipping session sampling.")
    except Exception as e:
        print(f"Error during session sampling process: {e}")
        traceback.print_exc()
else:
    print("Warning: No sessions were created or target sample size is 0, skipping session sampling.")

logon_output_file = output_dir / "filtered_logon_sessions.csv"
try:
    sampled_sessions_df.to_csv(logon_output_file, index=False)
    print(f"Total downsampled sessions: {len(sampled_sessions_df)}")
    print(f"File saved: {logon_output_file}")
except Exception as e:
    print(f"Error saving sampled sessions file {logon_output_file}: {e}")
    traceback.print_exc()

# --- Step 1.6: Filtering Activity Logs (Optimized Dask Approach) ---
print("\nStep 1.6: Filtering Activity Logs (Optimized Dask Approach)")

if not sampled_sessions_df.empty:
    # Convert sampled sessions to Dask DataFrame with a reasonable number of partitions
    ddf_sampled_sessions = dd.from_pandas(sampled_sessions_df, npartitions=max(1, os.cpu_count()))
    print(f"Created Dask DataFrame from sampled sessions with {ddf_sampled_sessions.npartitions} partitions.")

    # Ensure datetime columns are in the correct format
    ddf_sampled_sessions['logon_time'] = dd.to_datetime(ddf_sampled_sessions['logon_time'], errors='coerce')
    ddf_sampled_sessions['logoff_time'] = dd.to_datetime(ddf_sampled_sessions['logoff_time'], errors='coerce')
    ddf_sampled_sessions = ddf_sampled_sessions.dropna(subset=['logon_time', 'logoff_time'])

    activity_files = ['email.csv', 'file.csv', 'http.csv', 'device.csv']

    for file in activity_files:
        activity_file_path = base_data_dir / file
        filtered_output_path = output_dir / f"filtered_{file}"

        if not activity_file_path.exists():
            print(f"File {activity_file_path} not found. Skipping...")
            continue

        print(f"\nProcessing {file}...")
        try:
            # Define dtypes for activity files
            activity_dtypes_map = {
                'email.csv': {'id': 'str', 'date': 'str', 'user': 'str', 'pc': 'str', 'activity': 'str', 'to': 'str', 'cc': 'str', 'from': 'str', 'subject': 'str', 'path': 'str', 'filename': 'str'},
                'file.csv': {'id': 'str', 'date': 'str', 'user': 'str', 'pc': 'str', 'activity': 'str', 'filename': 'str', 'destination': 'str'},
                'http.csv': {'id': 'str', 'date': 'str', 'user': 'str', 'pc': 'str', 'activity': 'str', 'url': 'str', 'content': 'str'},
                'device.csv': {'id': 'str', 'date': 'str', 'user': 'str', 'pc': 'str', 'activity': 'str', 'logon_id': 'str', 'target_pc': 'str'}
            }
            current_activity_dtypes = activity_dtypes_map.get(file, {})
            core_dtypes = {'id': 'str', 'date': 'str', 'user': 'str', 'pc': 'str', 'activity': 'str'}
            activity_dtypes = {**core_dtypes, **current_activity_dtypes}

            # Load activity file with Dask
            ddf_activity = dd.read_csv(
                activity_file_path,
                dtype=activity_dtypes,
                blocksize='128MB',
                assume_missing=True,
                on_bad_lines='warn'
            )

            if 'user' not in ddf_activity.columns or 'date' not in ddf_activity.columns:
                print(f"Warning: Essential columns ('user' or 'date') not found in {file}. Skipping this file.")
                continue

            # Filter for selected users and parse dates
            ddf_activity = ddf_activity[ddf_activity['user'].isin(selected_users_list)]
            ddf_activity['date'] = dd.to_datetime(ddf_activity['date'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
            ddf_activity = ddf_activity.dropna(subset=['date', 'user'])
            print(f"Activity rows after user filter & date parsing (estimated): {len(ddf_activity)}")

            # Perform a single merge across all users
            ddf_merged = dd.merge(
                ddf_activity,
                ddf_sampled_sessions[['user', 'logon_time', 'logoff_time']],
                on='user',
                how='inner'
            )

            # Filter activities within session time windows
            ddf_merged = ddf_merged.dropna(subset=['date', 'logon_time', 'logoff_time'])
            ddf_filtered = ddf_merged[
                (ddf_merged['date'] >= ddf_merged['logon_time']) &
                (ddf_merged['date'] <= ddf_merged['logoff_time'])
            ]

            # Keep only the original activity columns
            original_activity_columns = list(ddf_activity.columns)
            ddf_final_activity = ddf_filtered[original_activity_columns]

            # Remove duplicates by 'id' if present
            if 'id' in ddf_final_activity.columns:
                ddf_final_activity = ddf_final_activity.set_index('id').drop_duplicates().reset_index()

            # Compute the final result once
            print(f"Computing filtered activities for {file}...")
            filtered_activity_df = ddf_final_activity.compute()
            print(f"Total filtered activities for {file}: {len(filtered_activity_df)}")

            # Save to CSV (removed single_file argument since this is a Pandas DataFrame)
            filtered_output_path.mkdir(parents=True, exist_ok=True)
            output_file_csv = filtered_output_path / f"filtered_{file}"
            filtered_activity_df.to_csv(output_file_csv, index=False)
            print(f"Finished saving filtered {file} to {output_file_csv}.")

        except Exception as e:
            print(f"Error processing {file}: {e}")
            traceback.print_exc()
else:
    print("Skipping activity log filtering as no sampled sessions were generated.")

# --- Summary ---
print("\n--- Processing Summary ---")
print(f"LDAP: {len(selected_users)}")
if 'psychometric_df_filtered' in locals() and psychometric_output_file.exists() and not psychometric_df_filtered.empty:
    print(f"Psychometric: {len(psychometric_df_filtered)} rows")
else:
    print("Psychometric: File not processed or empty.")
print(f"Logon (Sampled Sessions): {len(sampled_sessions_df)} rows")
print(f"Filtered Activities: Saved as CSV files in subdirectories within {output_dir}.")
print(f"Target Session Sample Size per user: {n_sessions_target}")
print(f"Current time: {datetime.now()}")
print("--- End Summary ---")