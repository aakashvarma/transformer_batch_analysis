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
    parser.add_argument('--min_h', type=int, default=None,
                        help='Minimum hidden size to include (default: None)')
    parser.add_argument('--max_h', type=int, default=None,
                        help='Maximum hidden size to include (default: None)')
    parser.add_argument('--min_bs', type=int, default=None,
                        help='Minimum batch size to include (default: None)')
    parser.add_argument('--max_bs', type=int, default=None,
                        help='Maximum batch size to include (default: None)')
    return parser.parse_args()

def load_and_process_data(csv_path, min_h=None, max_h=None, min_bs=None, max_bs=None):
    """
    Load and process the CSV file with filtering options
    """
    df = pd.read_csv(csv_path)
    
    # Create dense_init series for seqlen=1
    dense_init_mask = (df['series'] == 'dense') & (df['seqlen'] == 1)
    df.loc[dense_init_mask, 'series'] = 'dense_init'
    
    # Rename remaining dense to dense_ar for consistency
    df.loc[df['series'] == 'dense', 'series'] = 'dense_ar'
    
    # Apply filters if specified
    if min_h is not None:
        df = df[df['h'] >= min_h]
    if max_h is not None:
        df = df[df['h'] <= max_h]
    if min_bs is not None:
        df = df[df['bs'] >= min_bs]
    if max_bs is not None:
        df = df[df['bs'] <= max_bs]
    
    return df

def plot_roofline_model(df, output_dir):
    """
    Create roofline model plots with y-axis starting from 0
    """
    def format_label(x):
        if x >= 1e12:
            return f'{int(x/1e12)} T'
        elif x >= 1e9:
            return f'{int(x/1e9)} G'
        elif x >= 1e6:
            return f'{int(x/1e6)} M'
        elif x >= 1e3:
            return f'{int(x/1e3)} k'
        return str(int(x))

    # Define peak performance constants at the top
    peak_memory_bandwidth = 1.935e12  # 1.935 TB/s
    peak_compute = 312e12  # 312 TFLOP/s

    for dtype in df['dtype'].unique():
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Get data for current dtype
        dtype_data = df[df['dtype'] == dtype]
        
        # Define colors for different series
        colors = {
            'dense_init': '#1f77b4',  # blue
            'dense_ar': '#ff7f0e',    # orange
            'qk_init': '#2ca02c',     # green
            'qk_ar': '#d62728'        # red
        }
        
        # Calculate axis limits from data
        min_intensity = 1  # start x-axis from 1
        max_intensity = max(dtype_data['intensity'].max() * 1.2, 10000)  # extend to at least 10000
        max_flop = max(dtype_data['flop'].max() * 1.2, peak_compute)

        # Add roofline boundaries
        x = np.logspace(0, np.log10(max_intensity), 1000)  # start from 10^0 = 1
        memory_bound = peak_memory_bandwidth * x
        compute_bound = np.full_like(x, peak_compute)
        
        # Plot roofline
        plt.plot(x, np.minimum(memory_bound, compute_bound), 'k:', linewidth=1)

        # Plot points for each series
        series_order = ['dense_init', 'dense_ar', 'qk_init', 'qk_ar']
        for series in series_order:
            series_data = dtype_data[dtype_data['series'] == series]
            if len(series_data) > 0:
                for h_val in sorted(series_data['h'].unique()):
                    h_data = series_data[series_data['h'] == h_val]
                    alpha = 0.3 if h_val == 4096 else 0.6
                    label = f'{series} (h={h_val})'
                    plt.scatter(h_data['intensity'], h_data['flop'],
                              color=colors[series], alpha=alpha, s=30,
                              label=label)

        plt.xscale('log')
        plt.yscale('log')
        
        # Set axis limits
        plt.xlim(1, max_intensity)
        plt.ylim(1e5, max_flop)  # start from a very small value to show near-zero

        # Customize x-axis labels
        x_ticks = [1, 10, 100, 1000, 10000]
        plt.xticks(x_ticks, [str(x) for x in x_ticks])

        # Customize y-axis labels with more granular steps
        y_ticks = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
        y_labels = ['100 k', '1 M', '10 M', '100 M', '1 G', '10 G', '100 G', '1 T', '10 T', '100 T', '1000 T']
        plt.yticks(y_ticks, y_labels)
        
        # Add grid with specific style
        ax.grid(True, which="major", ls="-", alpha=0.2, color='lightgray')
        ax.grid(True, which="minor", ls="-", alpha=0.1, color='lightgray')
        
        # Make sure grid is behind points
        ax.set_axisbelow(True)
        
        plt.xlabel('FLOP : I/O')
        plt.ylabel('Achieved FLOP/s')
        plt.title(f'Roofline Model for {dtype}')

        # Create legend
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color='k', linestyle=':'))
        handles.append(plt.Line2D([0], [0], color='k', linestyle=':'))
        labels.append('1.935TB/s')
        labels.append('312 TFLOP/s')

        # Adjust legend style
        plt.legend(handles=handles, labels=labels,
                  bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=10, framealpha=1)

        # Adjust layout
        plt.tight_layout()
        
        # Save with high quality
        plt.savefig(os.path.join(output_dir, f'roofline_model_{dtype}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()

def plot_throughput_vs_flops_per_series_dtype(df, min_seqlen, output_dir):
    """
    Create separate throughput vs FLOPs plots for each series and dtype
    """
    series_order = ['dense_init', 'dense_ar', 'qk_init', 'qk_ar']
    for series in series_order:
        for dtype in df['dtype'].unique():
            plt.figure(figsize=(12, 8))
            
            # For dense_ar, filter by min_seqlen; for others, use all data
            if series == 'dense_ar':
                data = df[(df['dtype'] == dtype) & 
                         (df['series'] == series) & 
                         (df['seqlen'] >= min_seqlen)]
            else:
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
            title = f'{series.upper()} Throughput vs FLOPs for {dtype}'
            if series == 'dense_ar':
                title += f'\n(Sequence Length ≥ {min_seqlen})'
            plt.title(title)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{series}_throughput_vs_flops_{dtype}.png'), bbox_inches='tight')
            plt.close()

def plot_arithmetic_intensity_per_series_dtype(df, min_seqlen, output_dir):
    """
    Create separate arithmetic intensity analysis plots for each series and dtype
    """
    series_order = ['dense_init', 'dense_ar', 'qk_init', 'qk_ar']
    for series in series_order:
        for dtype in df['dtype'].unique():
            plt.figure(figsize=(12, 8))
            
            # For dense_ar, filter by min_seqlen; for others, use all data
            if series == 'dense_ar':
                data = df[(df['dtype'] == dtype) & 
                         (df['series'] == series) & 
                         (df['seqlen'] >= min_seqlen)]
            else:
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
            title = f'{series.upper()} Roofline Analysis: Throughput vs Arithmetic Intensity for {dtype}'
            if series == 'dense_ar':
                title += f'\n(Sequence Length ≥ {min_seqlen})'
            plt.title(title)
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{series}_arithmetic_intensity_{dtype}.png'), bbox_inches='tight')
            plt.close()

def plot_latency_vs_seqlen_per_series_dtype(df, min_seqlen, output_dir):
    """
    Create separate latency vs sequence length plots for each series and dtype
    """
    series_order = ['dense_init', 'dense_ar', 'qk_init', 'qk_ar']
    for series in series_order:
        for dtype in df['dtype'].unique():
            plt.figure(figsize=(12, 8))
            
            # For dense_ar, filter by min_seqlen; for others, use all data
            if series == 'dense_ar':
                data = df[(df['dtype'] == dtype) & 
                         (df['series'] == series) & 
                         (df['seqlen'] >= min_seqlen)]
            else:
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
            title = f'{series.upper()} Latency vs Sequence Length for {dtype}'
            if series == 'dense_ar':
                title += f'\n(Sequence Length ≥ {min_seqlen})'
            plt.title(title)
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{series}_latency_vs_seqlen_{dtype}.png'), bbox_inches='tight')
            plt.close()

def plot_batch_size_impact_per_series_dtype(df, min_seqlen, output_dir):
    """
    Create separate batch size impact analysis plots for each series and dtype
    """
    series_order = ['dense_init', 'dense_ar', 'qk_init', 'qk_ar']
    for series in series_order:
        for dtype in df['dtype'].unique():
            plt.figure(figsize=(12, 8))
            
            # For dense_ar, filter by min_seqlen; for others, use all data
            if series == 'dense_ar':
                data = df[(df['dtype'] == dtype) & 
                         (df['series'] == series) & 
                         (df['seqlen'] >= min_seqlen)]
            else:
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
            title = f'{series.upper()} Throughput vs Batch Size for {dtype}'
            if series == 'dense_ar':
                title += f'\n(Sequence Length ≥ {min_seqlen})'
            plt.title(title)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{series}_batch_size_impact_{dtype}.png'), bbox_inches='tight')
            plt.close()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set default plot style
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # Load data with filters
    print(f"Loading data from {args.csv_path}")
    print(f"Applying filters:")
    if args.min_seqlen > 1:
        print(f"  - Minimum sequence length: {args.min_seqlen}")
    if args.min_h:
        print(f"  - Minimum hidden size: {args.min_h}")
    if args.max_h:
        print(f"  - Maximum hidden size: {args.max_h}")
    if args.min_bs:
        print(f"  - Minimum batch size: {args.min_bs}")
    if args.max_bs:
        print(f"  - Maximum batch size: {args.max_bs}")
        
    df = load_and_process_data(
        args.csv_path,
        min_h=args.min_h,
        max_h=args.max_h,
        min_bs=args.min_bs,
        max_bs=args.max_bs
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create all plots
    print("Generating plots...")
    plot_roofline_model(df, args.output_dir)
    plot_throughput_vs_flops_per_series_dtype(df, args.min_seqlen, args.output_dir)
    plot_latency_vs_seqlen_per_series_dtype(df, args.min_seqlen, args.output_dir)
    plot_arithmetic_intensity_per_series_dtype(df, args.min_seqlen, args.output_dir)
    plot_batch_size_impact_per_series_dtype(df, args.min_seqlen, args.output_dir)
    print("\nSummary Statistics by dtype and series:")
    for dtype in df['dtype'].unique():
        print(f"\nDtype: {dtype}")
        series_order = ['dense_init', 'dense_ar', 'qk_init', 'qk_ar']
        for series in series_order:
            subset = df[(df['dtype'] == dtype) & (df['series'] == series)]
            if len(subset) > 0:
                print(f"\n{series.upper()}:")
                print(f"Average Throughput: {subset['throughput'].mean():,.2f} Tokens/s")
                print(f"Average Latency: {subset['latency'].mean():,.2f} ms")
                print(f"Maximum Throughput: {subset['throughput'].max():,.2f} Tokens/s")
                print(f"Minimum Latency: {subset['latency'].min():,.2f} ms")
    
    print(f"\nPlots have been saved to {args.output_dir}")

if __name__ == "__main__":
    main()



    # Show all data
# python plot.py data.csv --output_dir ./plots

# # Filter specific ranges
# python plot.py data.csv --output_dir ./plots --min_h 1024 --max_h 4096 --min_bs 1 --max_bs 128 --min_seqlen 2

# # Focus on specific h value
# python plot.py data.csv --output_dir ./plots --min_h 4096 --max_h 4096