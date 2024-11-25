import os
import gzip
import pickle
import pandas as pd
from glob import glob
from datetime import datetime

def read_gzip_pickle(filepath):
    """Read a gzipped pickle file and return its contents."""
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)

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
    # Get all gzip files in the data directory
    data_files = glob('data/*.pkl.gz')
    if not data_files:
        print("No .pkl.gz files found in the data directory!")
        return
    
    # Process each gzip file
    for file_path in data_files:
        print(f"Processing {file_path}...")
        
        try:
            # Extract timestamp, gpu, and dtype info from filename
            # Expected format: YYYYMMDD-transformer-batching-{gpu}-{dtype}.pkl.gz
            filename = os.path.basename(file_path)
            parts = filename.split('-')
            timestamp = parts[0]
            gpu_name = parts[-2]
            dtype = parts[-1].replace('.pkl.gz', '')
            
            # Read and process the data
            data = read_gzip_pickle(file_path)
            results_df = process_benchmark_data(data, dtype)
            
            # Add GPU information
            results_df['gpu'] = gpu_name
            
            # Generate output filename following original format:
            # transformer-batching-microbenchmarks-{gpu-name}-multi-dtype-{timestamp}.csv
            output_file = f"data/transformer-batching-microbenchmarks-{gpu_name}-multi-dtype-{timestamp}.csv"
            
            # Save to CSV
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
            print(f"Total records: {len(results_df)}")
            
            # Print summary for this file
            print(f"\nSummary of processed data for {gpu_name} {dtype}:")
            print("Operation types:", results_df['series'].unique())
            print("Number of different batch sizes:", len(results_df['bs'].unique()))
            print("Number of different sequence lengths:", len(results_df['seqlen'].unique()))
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()