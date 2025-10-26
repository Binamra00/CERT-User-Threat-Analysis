import dask.dataframe as dd
import pathlib
import os

# --- Configuration ---
base_data_dir = pathlib.Path(
    r"")

# List of files to process
files_to_count = ['email.csv', 'http.csv', 'device.csv', 'file.csv']

# --- Count Rows for Each File ---
print("Counting rows in r4.2 dataset files...\n")

for file in files_to_count:
    file_path = base_data_dir / file
    if not file_path.exists():
        print(f"File {file_path} not found. Skipping...")
        continue

    try:
        # Read the CSV file with Dask
        ddf = dd.read_csv(
            file_path,
            blocksize='512MB',  # Use a reasonable blocksize to manage memory
            assume_missing=True,
            on_bad_lines='warn'  # Skip bad lines to avoid errors
        )

        # Compute the number of rows
        num_rows = len(ddf)
        print(f"Number of rows in {file}: {num_rows:,}")

    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

print("\nRow counting completed.")