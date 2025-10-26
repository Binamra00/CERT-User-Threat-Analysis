import dask.dataframe as dd
import pandas as pd
import os
import pathlib
import traceback
import dask
import tempfile
from tqdm import tqdm

# --- Configure Dask Temporary Directory ---
# Define the desired temporary directory on C: drive
desired_temp_dir = 'C:/temp'

# Check if the desired directory exists, create it if it doesn't
try:
    pathlib.Path(desired_temp_dir).mkdir(parents=True, exist_ok=True)
    print(f"Temporary directory {desired_temp_dir} is ready for use.")
except Exception as e:
    print(f"Warning: Could not create temporary directory {desired_temp_dir}: {e}")
    # Fallback to system default temporary directory
    desired_temp_dir = tempfile.gettempdir()
    print(f"Falling back to system default temporary directory: {desired_temp_dir}")

# Configure Dask to use the temporary directory
dask.config.set({'temporary_directory': desired_temp_dir, 'array.chunk-size': '256MiB'})
dask.config.set({'distributed.scheduler.work-stealing': False})

# --- Configuration ---
base_data_dir = pathlib.Path(r"F:\UTA Graduate Studies\Semester 1\CSE 5301 Data Analysis and Modeling Techniques\Group Project\Dataset\r4.2")
output_dir = base_data_dir / "filtered_output"

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# --- Step 1: Load Filtered LDAP Users ---
print("Step 1: Loading Filtered LDAP Users")
ldap_output_file = output_dir / "filtered_ldap.csv"
if not ldap_output_file.exists():
    print(f"Error: {ldap_output_file} not found. Please ensure the file exists from previous steps.")
    exit()

try:
    ldap_deduped = pd.read_csv(ldap_output_file, dtype={'user_id': 'str'})
    selected_users = set(ldap_deduped['user_id'])
    selected_users_list = list(selected_users)
    print(f"Loaded {len(selected_users)} LDAP users from {ldap_output_file}")
except Exception as e:
    print(f"Error loading LDAP file {ldap_output_file}: {e}")
    traceback.print_exc()
    exit()

# --- Step 2: Load Downsampled Sessions ---
print("\nStep 2: Loading Downsampled Sessions")
logon_output_file = output_dir / "filtered_logon_sessions.csv"
if not logon_output_file.exists():
    print(f"Error: {logon_output_file} not found. Please ensure the file exists from previous steps.")
    exit()

try:
    sampled_sessions_df = pd.read_csv(logon_output_file)
    sampled_sessions_df['logon_time'] = pd.to_datetime(sampled_sessions_df['logon_time'])
    sampled_sessions_df['logoff_time'] = pd.to_datetime(sampled_sessions_df['logoff_time'])
    print(f"Loaded {len(sampled_sessions_df)} downsampled sessions from {logon_output_file}")
except Exception as e:
    print(f"Error loading downsampled sessions file {logon_output_file}: {e}")
    traceback.print_exc()
    exit()

# --- Step 3: Filtering http.csv (Session-Based Filtering with Batch Processing) ---
print("\nStep 3: Filtering http.csv (Session-Based Filtering with Batch Processing)")

if not sampled_sessions_df.empty:
    file = 'http.csv'
    activity_file_path = base_data_dir / file
    filtered_output_path = output_dir / f"filtered_{file}"

    if not activity_file_path.exists():
        print(f"File {activity_file_path} not found. Exiting...")
        exit()

    print(f"\nProcessing {file}...")
    try:
        # Define dtypes for http.csv
        activity_dtypes = {
            'id': 'str',
            'date': 'str',
            'user': 'str',
            'pc': 'str',
            'activity': 'str',
            'url': 'str',
            'content': 'str'
        }

        # Load http.csv with Dask, using a larger blocksize to reduce partitions
        blocksize = '512MB'
        ddf_activity = dd.read_csv(
            activity_file_path,
            dtype=activity_dtypes,
            blocksize=blocksize,
            assume_missing=True,
            on_bad_lines='warn'
        )

        if 'user' not in ddf_activity.columns or 'date' not in ddf_activity.columns:
            print(f"Error: Essential columns ('user' or 'date') not found in {file}. Exiting...")
            exit()

        # Filter for selected users and parse dates
        ddf_activity = ddf_activity[ddf_activity['user'].isin(selected_users_list)]
        ddf_activity['date'] = dd.to_datetime(ddf_activity['date'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
        ddf_activity = ddf_activity.dropna(subset=['date', 'user'])
        print(f"Activity rows after user filter & date parsing (estimated): {len(ddf_activity)}")

        # Repartition to reduce number of partitions
        target_partitions = max(1, os.cpu_count())
        ddf_activity = ddf_activity.repartition(npartitions=target_partitions)
        # Persist the filtered DataFrame to avoid recomputation
        ddf_activity = ddf_activity.persist()

        # Process users in batches
        batch_size = 100  # Number of users per batch
        all_filtered_dfs = []
        user_batches = [selected_users_list[i:i + batch_size] for i in range(0, len(selected_users_list), batch_size)]

        for batch_idx, user_batch in enumerate(tqdm(user_batches, desc="Processing user batches")):
            print(f"\nProcessing batch {batch_idx + 1}/{len(user_batches)} with {len(user_batch)} users...")

            # Filter activities and sessions for the current batch of users
            ddf_activity_batch = ddf_activity[ddf_activity['user'].isin(user_batch)]
            ddf_sampled_sessions_batch = dd.from_pandas(
                sampled_sessions_df[sampled_sessions_df['user'].isin(user_batch)],
                npartitions=max(1, os.cpu_count())
            )

            # Ensure datetime columns are in the correct format
            ddf_sampled_sessions_batch['logon_time'] = dd.to_datetime(ddf_sampled_sessions_batch['logon_time'], errors='coerce')
            ddf_sampled_sessions_batch['logoff_time'] = dd.to_datetime(ddf_sampled_sessions_batch['logoff_time'], errors='coerce')
            ddf_sampled_sessions_batch = ddf_sampled_sessions_batch.dropna(subset=['logon_time', 'logoff_time'])

            # Perform the merge for the current batch
            ddf_merged = dd.merge(
                ddf_activity_batch,
                ddf_sampled_sessions_batch[['user', 'logon_time', 'logoff_time']],
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

            # Compute the filtered activities for this batch
            print(f"Computing filtered activities for batch {batch_idx + 1}...")
            filtered_activity_df = ddf_final_activity.compute()
            print(f"Filtered activities in batch {batch_idx + 1}: {len(filtered_activity_df)}")

            # Collect the batch result
            all_filtered_dfs.append(filtered_activity_df)

        # Concatenate all batch results
        print("\nConcatenating all batch results...")
        if all_filtered_dfs:
            final_filtered_df = pd.concat(all_filtered_dfs, ignore_index=True)
        else:
            final_filtered_df = pd.DataFrame(columns=ddf_activity.columns)

        print(f"Total filtered activities for {file}: {len(final_filtered_df)}")

        # Save to CSV
        filtered_output_path.mkdir(parents=True, exist_ok=True)
        output_file_csv = filtered_output_path / f"filtered_{file}"
        final_filtered_df.to_csv(output_file_csv, index=False)
        print(f"Finished saving filtered {file} to {output_file_csv}.")

    except Exception as e:
        print(f"Error processing {file}: {e}")
        traceback.print_exc()
        exit()
else:
    print("Error: No sampled sessions available. Exiting...")
    exit()

# --- Summary ---
print("\n--- Processing Summary for http.csv ---")
print(f"Loaded LDAP Users: {len(selected_users)}")
print(f"Loaded Downsampled Sessions: {len(sampled_sessions_df)}")
print(f"Filtered http.csv: Saved to {output_dir / 'filtered_http.csv'}")
print("--- End Summary ---")