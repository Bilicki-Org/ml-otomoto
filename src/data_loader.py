import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy Azure SDK logs (only show warnings or errors)
logging.getLogger("azure").setLevel(logging.WARNING)

def download_single_blob(connection_string: str, container_name: str, blob_name: str, target_file: Path) -> bool:
    """
    Helper function to download a single blob. Designed to be run in a separate thread.
    
    Args:
        connection_string (str): Azure connection string.
        container_name (str): Name of the container.
        blob_name (str): Name of the specific file in the cloud.
        target_file (Path): Local path where the file should be saved.

    Returns:
        bool: True if download (or cache hit) was successful, False otherwise.
    """
    # 1. Check Local Cache
    if target_file.exists():
        return True  # Skip download, file already exists

    try:
        # Re-create client inside thread (thread-safe approach)
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # Download data
        with open(target_file, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        return True
    except Exception as e:
        logger.error(f"Failed to download {blob_name}: {e}")
        return False

def load_and_merge_data(local_dir: str = "data/raw", container_name: str = "raw-data") -> Path:
    """
    Main pipeline function.
    1. Connects to Azure.
    2. Lists all CSV batch files.
    3. Downloads them in parallel (multi-threaded) to a local 'batches' folder.
    4. Merges all downloaded batches into a single CSV dataset.

    Args:
        local_dir (str): Base directory for data storage.
        container_name (str): Azure Blob Storage container name.

    Returns:
        Path: The path to the final merged dataset.
    """
    # 1. Environment Setup
    load_dotenv()
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    
    if not connect_str:
        raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING in environment variables.")

    # 2. Directory Setup
    base_dir = Path(local_dir)
    batches_dir = base_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True) # Create data/raw/batches if not exists

    logger.info("🔌 Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)

    # 3. List Blobs
    logger.info("Listing files in container (this might take a moment)...")
    blobs = list(container_client.list_blobs())
    csv_blobs = [b for b in blobs if b.name.endswith('.csv')]
    
    if not csv_blobs:
        raise FileNotFoundError(f"No CSV files found in container '{container_name}'.")

    logger.info(f"Found {len(csv_blobs)} CSV batch files. Starting parallel download...")

    # 4. Parallel Execution (Multi-threading)
    # We use max_workers=10 to download 10 files simultaneously
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for blob in csv_blobs:
            target_path = batches_dir / blob.name
            
            # Submit task to the thread pool
            futures.append(executor.submit(
                download_single_blob, 
                connect_str, 
                container_name, 
                blob.name, 
                target_path
            ))
        
        # Display progress bar while tasks complete
        for _ in tqdm(as_completed(futures), total=len(futures), desc="⬇Downloading Batches", unit="file"):
            pass # We just wait for completion here

    # 5. Merge Data
    logger.info("🔄 Merging downloaded batches into a single dataset...")
    
    all_batch_files = list(batches_dir.glob("*.csv"))
    if not all_batch_files:
        raise FileNotFoundError("No files found locally to merge.")

    dataframes_list = []
    
    # Read each file and append to list
    for file_path in tqdm(all_batch_files, desc="📖 Reading CSVs", unit="file"):
        try:
            # on_bad_lines='skip' ensures one bad file doesn't crash the whole pipeline
            df = pd.read_csv(file_path, on_bad_lines='skip')
            dataframes_list.append(df)
        except Exception as e:
            logger.warning(f"Could not read file {file_path.name}: {e}")

    if not dataframes_list:
        raise ValueError("No valid dataframes loaded.")

    # Concatenate all mini-dataframes into one big dataframe
    merged_df = pd.concat(dataframes_list, ignore_index=True)
    
    # 6. Save Final Result
    output_path = base_dir / "otomoto_all_data.csv"
    merged_df.to_csv(output_path, index=False)
    
    logger.info("Pipeline finished successfully!")
    logger.info(f"Final Dataset Stats: {len(merged_df)} rows saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    try:
        load_and_merge_data()
    except Exception as e:
        logger.error(f"Critical Failure: {e}")