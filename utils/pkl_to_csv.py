import os
import gzip
import pickle
import pandas as pd
import argparse
from glob import glob
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process transformer batching benchmark data.')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing the benchmark .pkl.gz files')
    return parser.parse_args()

def read_gzip_pickle(filepath):
    """Read a gzipped pickle file and return its contents."""
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)

def extract_file_info(filename):
    """Extract timestamp, GPU name, and dtype from filename."""
    # Example: 20241125-transformer-batching-h100-bf16.pkl.gz
    parts = filename.split('-')
    timestamp = parts[0]
    gpu_name = parts[-2]
    dtype = parts[-1].replace('.pkl.gz', '')
    return timestamp, gpu_name, dtype

def process_benchmark_data(data, dtype_name):
    """Process benchmark data for a single dtype into DataFrames."""
    # Process dense operations
    df_dense = (
        pd.DataFrame.from_dict(data["dense"])
        .assign(h=lambda x: x["n"] * x["d"])
        .assign(flop=lambda x: (x["bs"] * x["seqlen"] * x["h"]**2) * 2)
        .assign(io=lambda x: (x["bs"]*x["seqlen"]*x["h"]*2 + x["h"]**2) * 2)
        .assign(intensity=lambda x: x["flop"] / x["io"])
        .assign(throughput=lambda x: x["bs"]*x["seqlen"] / x["latency"])
        .assign(series="dense")
        .assign(dtype=dtype_name)
    )
    
    # Process QK initialization
    df_qk_init = (
        pd.DataFrame.from_dict(data["qk_init"])
        .assign(h=lambda x: x["n"] * x["d"])
        .assign(flop=lambda x: (x["bs"]*x["n"]*x["d"]*x["seqlen"]**2) * 2)
        .assign(io=lambda x: (x["bs"]*x["n"]*(x["seqlen"]*x["d"]*2 + x["seqlen"]**2)) * 2)
        .assign(intensity=lambda x: x["flop"] / x["io"])
        .assign(throughput=lambda x: x["bs"]*x["seqlen"] / x["latency"])
        .assign(series="qk_init")
        .assign(dtype=dtype_name)
    )
    
    # Process QK autoregressive
    df_qk_ar = (
        pd.DataFrame.from_dict(data["qk_ar"])
        .assign(h=lambda x: x["n"] * x["d"])
        .assign(flop=lambda x: (x["bs"]*x["n"]*x["d"]*x["seqlen"]) * 2)
        .assign(io=lambda x: (x["bs"]*x["n"]*(x["d"] + x["seqlen"]*x["d"] + x["seqlen"])) * 2)
        .assign(intensity=lambda x: x["flop"] / x["io"])
        .assign(throughput=lambda x: x["bs"] / x["latency"])
        .assign(series="qk_ar")
        .assign(dtype=dtype_name)
    )
    
    return pd.concat([df_dense, df_qk_init, df_qk_ar])

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get all gzip files in the data directory
    data_files = glob(os.path.join(args.data_dir, '*.pkl.gz'))
    if not data_files:
        print(f"No .pkl.gz files found in the directory: {args.data_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.data_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store DataFrames from all files
    all_results = []
    gpu_name = None
    timestamp = None
    
    # Process each gzip file
    for file_path in data_files:
        print(f"Processing {file_path}...")
        
        try:
            # Extract file information
            filename = os.path.basename(file_path)
            curr_timestamp, curr_gpu, dtype = extract_file_info(filename)
            
            # Store timestamp and GPU name from first file
            if timestamp is None:
                timestamp = curr_timestamp
            if gpu_name is None:
                gpu_name = curr_gpu
            
            # Verify files are from same batch
            if curr_timestamp != timestamp or curr_gpu != gpu_name:
                print(f"Warning: File {filename} appears to be from a different batch")
                continue
            
            # Read and process the data
            data = read_gzip_pickle(file_path)
            results_df = process_benchmark_data(data, dtype)
            
            # Add GPU information
            results_df['gpu'] = gpu_name
            
            # Add to list of results
            all_results.append(results_df)
            
            print(f"Processed {dtype} data: {len(results_df)} records")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Generate output filename
        output_file = os.path.join(
            output_dir,
            f"transformer-batching-microbenchmarks-{gpu_name}-multi-dtype-{timestamp}.csv"
        )
        
        # Save to CSV
        combined_results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"Total records: {len(combined_results)}")
        
        # Print summary
        print("\nSummary of processed data:")
        print("Operation types:", combined_results['series'].unique())
        print("Data types:", combined_results['dtype'].unique())
        print("Number of different batch sizes:", len(combined_results['bs'].unique()))
        print("Number of different sequence lengths:", len(combined_results['seqlen'].unique()))
        
        # Print records per dtype
        print("\nRecords per dtype:")
        print(combined_results.groupby('dtype').size())
        
    else:
        print("No data was successfully processed")

if __name__ == "__main__":
    main()