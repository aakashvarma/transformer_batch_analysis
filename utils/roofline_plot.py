"""
Combined Roofline Plot Generator for A100 and H100 GPUs

Usage:
    python roofline_plot.py csv_path [options]

Required Arguments:
    csv_path            Path to the CSV file containing the data
                       CSV should have columns: n,d,seqlen,bs,dtype,latency,h,flop,io,intensity,throughput,series,gpu

Optional Arguments:
    --output-prefix    Prefix for output files (default: roofline)
    --seqlen          Sequence length to filter for
    --hidden-size     Hidden size (h) to filter for
    --min-bs         Minimum batch size to include
    --max-bs         Maximum batch size to include
    --series         List of series to include (space-separated)

Example Usage:
    # Basic usage with default settings
    python roofline_plot.py data.csv

    # Filtering for specific sequence length and series
    python roofline_plot.py data.csv --seqlen 512 --series dense qk_init

    # Setting batch size range and output prefix
    python roofline_plot.py data.csv --min-bs 8 --max-bs 32 --output-prefix my_roofline

    # Full example with all options
    python roofline_plot.py data.csv --seqlen 1024 --hidden-size 64 --min-bs 16 --max-bs 64 --series dense qk_init --output-prefix custom_roofline
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.1
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# GPU specifications
GPU_SPECS = {
    'a100': {
        'fp32': {'compute': 312, 'bandwidth': 1.935},
        'fp16': {'compute': 312, 'bandwidth': 1.935},
        'bf16': {'compute': 312, 'bandwidth': 1.935}
    },
    'h100': {
        'fp32': {'compute': 989, 'bandwidth': 3.35},
        'fp16': {'compute': 1979, 'bandwidth': 3.35},
        'bf16': {'compute': 1979, 'bandwidth': 3.35},
        'fp8': {'compute': 3958, 'bandwidth': 3.35},
        'int8': {'compute': 3958, 'bandwidth': 3.35}
    }
}

def format_func(value, pos):
    if value >= 1e12:
        return f'{value/1e12:.0f} T'
    elif value >= 1e9:
        return f'{value/1e9:.0f} G'
    elif value >= 1e6:
        return f'{value/1e6:.0f} M'
    return f'{value:.0f}'

def create_color_gradient(base_color, n_colors):
    return LinearSegmentedColormap.from_list("", ["white", base_color])(np.linspace(0.3, 1, n_colors))

def create_roofline_plot(df, gpu_type, dtype, seqlen=None, hidden_size=None, min_bs=None, max_bs=None, series_filter=None):
    # Get GPU specifications
    gpu_specs = GPU_SPECS.get(gpu_type.lower(), GPU_SPECS['a100'])
    peak_values = gpu_specs.get(dtype, gpu_specs['fp32'])
    peak_compute = peak_values['compute']
    peak_bandwidth = peak_values['bandwidth']
    
    # Apply filters
    if seqlen is not None:
        df = df[df['seqlen'] == seqlen]
    if hidden_size is not None:
        df = df[df['h'] == hidden_size]
    if min_bs is not None:
        df = df[df['bs'] >= min_bs]
    if max_bs is not None:
        df = df[df['bs'] <= max_bs]
    if series_filter:
        df = df[df['series'].isin(series_filter)]
    
    # Calculate achieved FLOP/s
    df['achieved_flops'] = df['flop'] / df['latency']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Series colors
    series_colors = {
        'dense': 'blue',
        'qk_init': 'green',
        'qk_ar': 'red'
    }
    
    # Plot data points
    legend_handles = []
    for series in sorted(df['series'].unique()):
        series_data = df[df['series'] == series]
        h_values = sorted(series_data['h'].unique())
        colors = create_color_gradient(series_colors.get(series, 'gray'), len(h_values))
        
        for i, h in enumerate(h_values):
            h_data = series_data[series_data['h'] == h]
            if not h_data.empty:
                scatter = plt.scatter(h_data['intensity'], 
                                    h_data['achieved_flops'],
                                    label=f'{series} (h={h})',
                                    color=colors[i],
                                    marker='o',
                                    alpha=0.8,
                                    s=30)
                legend_handles.append(scatter)
    
    if not df.empty:
        # Calculate ranges
        min_intensity = df['intensity'].min()
        max_intensity = df['intensity'].max()
        min_flops = df['achieved_flops'].min()
        max_flops = df['achieved_flops'].max()
        
        # Create roofline
        x = np.logspace(np.floor(np.log10(min_intensity)) - 0.5, 
                       np.ceil(np.log10(max_intensity)) + 0.5, 
                       1000)
        
        peak_compute_flops = peak_compute * 1e12
        peak_bandwidth_flops = peak_bandwidth * 1e12
        
        roofline = np.minimum(peak_compute_flops, peak_bandwidth_flops * x)
        line = plt.plot(x, roofline, ':',
                       color='#333333',
                       linewidth=1.5,
                       label=f'{gpu_type.upper()} {dtype}\n{peak_compute} TFLOP/s\n{peak_bandwidth} TB/s')[0]
        legend_handles.append(line)
        
        # Add auxiliary lines
        plt.axhline(y=peak_compute_flops, color='#333333', linestyle=':', alpha=0.2)
        plt.plot(x, peak_bandwidth_flops * x, color='#333333', linestyle=':', alpha=0.2)
        
        # Set scales and grid
        plt.xscale('log')
        plt.yscale('log')
        ax.grid(True, which="major", color='#E5E5E5', linestyle='-', alpha=0.5)
        ax.grid(True, which="minor", color='#E5E5E5', linestyle='-', alpha=0.2)
        ax.set_axisbelow(True)
        
        # Set limits
        plt.xlim(10**(np.floor(np.log10(min_intensity)) - 0.5),
                10**(np.ceil(np.log10(max_intensity)) + 0.5))
        plt.ylim(10**(np.floor(np.log10(min_flops)) - 0.5),
                10**(np.ceil(np.log10(max_flops)) + 0.5))
        
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        
        # Labels and title
        plt.xlabel('FLOP : I/O', fontsize=11, fontweight='bold', labelpad=10)
        plt.ylabel('Achieved FLOP/s', fontsize=11, fontweight='bold', labelpad=10)
        
        title = f'Roofline Plot - {gpu_type.upper()} {dtype.upper()}'
        if seqlen is not None:
            title += f'\nSequence Length = {seqlen}'
        plt.title(title, pad=20, fontweight='bold')
        
        plt.legend(handles=legend_handles, 
                  loc='center left', 
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=9,
                  framealpha=0.95)
        
        plt.tight_layout()
        return fig
    else:
        print(f"No data points match the specified filters for {gpu_type} {dtype}!")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate roofline plots for different GPUs and dtypes')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--output-prefix', type=str, default='roofline',
                       help='Prefix for output files (default: roofline)')
    parser.add_argument('--seqlen', type=int, help='Sequence length to filter for')
    parser.add_argument('--hidden-size', type=int, help='Hidden size (h) to filter for')
    parser.add_argument('--min-bs', type=int, help='Minimum batch size to include')
    parser.add_argument('--max-bs', type=int, help='Maximum batch size to include')
    parser.add_argument('--series', nargs='+', help='List of series to include')
    
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    
    # Create plots for each GPU and dtype combination in the data
    for gpu in df['gpu'].unique():
        df_gpu = df[df['gpu'] == gpu]
        for dtype in df_gpu['dtype'].unique():
            df_dtype = df_gpu[df_gpu['dtype'] == dtype]
            fig = create_roofline_plot(
                df_dtype,
                gpu_type=gpu,
                dtype=dtype,
                seqlen=args.seqlen,
                hidden_size=args.hidden_size,
                min_bs=args.min_bs,
                max_bs=args.max_bs,
                series_filter=args.series
            )
            
            if fig:
                output_file = f"{args.output_prefix}_{gpu}_{dtype}.png"
                fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Plot saved as {output_file}")
                plt.close(fig)

if __name__ == "__main__":
    main()