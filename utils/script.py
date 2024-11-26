import pandas as pd

def add_default_columns(input_file, output_file):
    """
    Read CSV file, add dtype and gpu columns with default values, and save to new CSV.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Add new columns with default values
    df['dtype'] = 'bf16'
    df['gpu'] = 'a100'
    
    # Reorder columns to match original structure
    columns_order = [
        'n', 'd', 'seqlen', 'bs', 'dtype', 'latency', 
        'h', 'flop', 'io', 'intensity', 'throughput', 
        'series', 'gpu'
    ]
    
    df = df[columns_order]
    
    # Save the modified dataframe to a new CSV file
    df.to_csv(output_file, index=False)
    
    return df

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_file = "/Users/aakashvarma/Documents/github/transformer_batch_analysis/data/processed/transformer-batching-microbenchmarks.csv"
    output_file = "/Users/aakashvarma/Documents/github/transformer_batch_analysis/data/processed/transformer-batching-microbenchmarks-a100-bf16-20241126"
    
    processed_df = add_default_columns(input_file, output_file)
    print("Processing complete. Added dtype=bf16 and gpu=a100 columns")