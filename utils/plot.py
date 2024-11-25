import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate performance analysis plots')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the plots (default: current directory)')
    parser.add_argument('--min_seqlen', type=int, default=2,
                        help='Minimum sequence length to include (default: 2)')
    return parser.parse_args()

def load_and_process_data(csv_path, min_seqlen):
    """
    Load and process the CSV file, filtering out low sequence lengths
    """
    df = pd.read_csv(csv_path)
    df = df[df['seqlen'] >= min_seqlen]
    return df

def plot_throughput_vs_flops_per_series_dtype(df, output_dir):
    """
    Create separate throughput vs FLOPs plots for each series and dtype
    """
    for series in df['series'].unique():
        for dtype in df['dtype'].unique():
            plt.figure(figsize=(12, 8))
            data = df[(df['dtype'] == dtype) & (df['series'] == series)]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(data['h'].unique())))
            
            for h_val, color in zip(sorted(data['h'].unique()), colors):
                subset = data[data['h'] == h_val]
                plt.scatter(subset['flop'], subset['throughput'],
                           label=f'h={h_val}', alpha=0.6, color=color)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('FLOPs')
            plt.ylabel('Throughput (Tokens/s)')
            plt.title(f'{series.upper()} Throughput vs FLOPs for {dtype}\n(Sequence Length ≥ {min_seqlen})')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{series}_throughput_vs_flops_{dtype}.png'), bbox_inches='tight')
            plt.close()

def plot_latency_vs_seqlen_per_series_dtype(df, output_dir):
    """
    Create separate latency vs sequence length plots for each series and dtype
    """
    for series in df['series'].unique():
        for dtype in df['dtype'].unique():
            plt.figure(figsize=(12, 8))
            data = df[(df['dtype'] == dtype) & (df['series'] == series)]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(data['h'].unique())))
            
            for h_val, color in zip(sorted(data['h'].unique()), colors):
                subset = data[data['h'] == h_val]
                plt.scatter(subset['seqlen'], subset['latency'],
                           label=f'h={h_val}', alpha=0.6, color=color)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Sequence Length')
            plt.ylabel('Latency (ms)')
            plt.title(f'{series.upper()} Latency vs Sequence Length for {dtype}\n(Sequence Length ≥ {min_seqlen})')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{series}_latency_vs_seqlen_{dtype}.png'), bbox_inches='tight')
            plt.close()

def plot_arithmetic_intensity_per_series_dtype(df, output_dir):
    """
    Create separate arithmetic intensity analysis plots for each series and dtype
    """
    for series in df['series'].unique():
        for dtype in df['dtype'].unique():
            plt.figure(figsize=(12, 8))
            data = df[(df['dtype'] == dtype) & (df['series'] == series)]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(data['h'].unique())))
            
            for h_val, color in zip(sorted(data['h'].unique()), colors):
                subset = data[data['h'] == h_val]
                plt.scatter(subset['intensity'], subset['throughput'],
                           label=f'h={h_val}', alpha=0.6, color=color)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
            plt.ylabel('Throughput (Tokens/s)')
            plt.title(f'{series.upper()} Roofline Analysis: Throughput vs Arithmetic Intensity for {dtype}\n(Sequence Length ≥ {min_seqlen})')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{series}_arithmetic_intensity_{dtype}.png'), bbox_inches='tight')
            plt.close()

def plot_batch_size_impact_per_series_dtype(df, output_dir):
    """
    Create separate batch size impact analysis plots for each series and dtype
    """
    for series in df['series'].unique():
        for dtype in df['dtype'].unique():
            plt.figure(figsize=(12, 8))
            data = df[(df['dtype'] == dtype) & (df['series'] == series)]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(data['h'].unique())))
            
            for seqlen in sorted(data['seqlen'].unique()):
                seq_data = data[data['seqlen'] == seqlen]
                
                for h_val, color in zip(sorted(seq_data['h'].unique()), colors):
                    subset = seq_data[seq_data['h'] == h_val]
                    plt.scatter(subset['bs'], subset['throughput'],
                              label=f'h={h_val}, seq={seqlen}', alpha=0.6, color=color)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (Tokens/s)')
            plt.title(f'{series.upper()} Throughput vs Batch Size for {dtype}\n(Sequence Length ≥ {min_seqlen})')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{series}_batch_size_impact_{dtype}.png'), bbox_inches='tight')
            plt.close()

def main():
    # Parse arguments
    args = parse_arguments()
    global min_seqlen
    min_seqlen = args.min_seqlen
    
    # Set default plot style
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # Load data
    print(f"Loading data from {args.csv_path}")
    print(f"Filtering out sequence lengths < {min_seqlen}")
    df = load_and_process_data(args.csv_path, min_seqlen)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create all plots
    print("Generating plots...")
    plot_throughput_vs_flops_per_series_dtype(df, args.output_dir)
    plot_latency_vs_seqlen_per_series_dtype(df, args.output_dir)
    plot_arithmetic_intensity_per_series_dtype(df, args.output_dir)
    plot_batch_size_impact_per_series_dtype(df, args.output_dir)
    
    # Print summary statistics
    print("\nSummary Statistics by dtype and series (for sequence length ≥ {min_seqlen}):")
    for dtype in df['dtype'].unique():
        print(f"\nDtype: {dtype}")
        for series in df['series'].unique():
            subset = df[(df['dtype'] == dtype) & (df['series'] == series)]
            print(f"\n{series.upper()}:")
            print(f"Average Throughput: {subset['throughput'].mean():,.2f} Tokens/s")
            print(f"Average Latency: {subset['latency'].mean():,.2f} ms")
            print(f"Maximum Throughput: {subset['throughput'].max():,.2f} Tokens/s")
            print(f"Minimum Latency: {subset['latency'].min():,.2f} ms")
    
    print(f"\nPlots have been saved to {args.output_dir}")

if __name__ == "__main__":
    main()