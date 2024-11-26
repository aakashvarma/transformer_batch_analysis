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

def create_roofline_plot(df, peak_compute, peak_bandwidth, seqlen=None, hidden_size=None, min_bs=None, max_bs=None, series_filter=None):
    # Apply filters if provided
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
    
    # Base colors for each series
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
        # Get data ranges for dynamic scaling
        min_intensity = df['intensity'].min()
        max_intensity = df['intensity'].max()
        min_flops = df['achieved_flops'].min()
        max_flops = df['achieved_flops'].max()
        
        # Create roofline with proper range
        x = np.logspace(np.floor(np.log10(min_intensity)) - 0.5, 
                       np.ceil(np.log10(max_intensity)) + 0.5, 
                       1000)
        
        # Convert peak compute to FLOP/s and bandwidth to TB/s
        peak_compute_flops = peak_compute * 1e12  # TFLOP/s to FLOP/s
        peak_bandwidth_flops = peak_bandwidth * 1e12  # TB/s to B/s
        
        # Calculate roofline
        roofline = np.minimum(peak_compute_flops, peak_bandwidth_flops * x)
        line = plt.plot(x, roofline, ':',
                       color='#333333',
                       linewidth=1.5,
                       label=f'{peak_compute} TFLOP/s, {peak_bandwidth} TB/s')[0]
        legend_handles.append(line)
        
        # Add horizontal line for compute bound
        plt.axhline(y=peak_compute_flops, color='#333333', linestyle=':', alpha=0.2)
        
        # Add diagonal line for memory bound
        memory_bound = peak_bandwidth_flops * x
        plt.plot(x, memory_bound, color='#333333', linestyle=':', alpha=0.2)
        
        plt.xscale('log')
        plt.yscale('log')
        
        # Set grid style
        ax.grid(True, which="major", color='#E5E5E5', linestyle='-', alpha=0.5)
        ax.grid(True, which="minor", color='#E5E5E5', linestyle='-', alpha=0.2)
        ax.set_axisbelow(True)
        
        # Set dynamic axis limits with padding
        plt.xlim(10**(np.floor(np.log10(min_intensity)) - 0.5),
                10**(np.ceil(np.log10(max_intensity)) + 0.5))
        plt.ylim(10**(np.floor(np.log10(min_flops)) - 0.5),
                10**(np.ceil(np.log10(max_flops)) + 0.5))
        
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        
        plt.xlabel('FLOP : I/O', fontsize=11, fontweight='bold', labelpad=10)
        plt.ylabel('Achieved FLOP/s', fontsize=11, fontweight='bold', labelpad=10)
        
        # Create title based on filters
        title = 'Roofline Model'
        if seqlen is not None:
            title += f' (seqlen={seqlen})'
        plt.title(title, pad=20, fontweight='bold')
        
        plt.legend(handles=legend_handles, 
                  loc='center left', 
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=9,
                  framealpha=0.95)
        
        plt.tight_layout()
        return fig
    else:
        print("No data points match the specified filters!")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate a roofline plot from CSV data')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--output', type=str, default='roofline_plot.png',
                       help='Output file path (default: roofline_plot.png)')
    parser.add_argument('--peak-compute', type=float, default=312,
                       help='Peak compute performance in TFLOP/s (default: 312)')
    parser.add_argument('--peak-bandwidth', type=float, default=1.935,
                       help='Peak memory bandwidth in TB/s (default: 1.935)')
    parser.add_argument('--seqlen', type=int, help='Sequence length to filter for')
    parser.add_argument('--hidden-size', type=int, help='Hidden size (h) to filter for')
    parser.add_argument('--min-bs', type=int, help='Minimum batch size to include')
    parser.add_argument('--max-bs', type=int, help='Maximum batch size to include')
    parser.add_argument('--series', nargs='+', help='List of series to include')
    
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    
    fig = create_roofline_plot(
        df,
        peak_compute=args.peak_compute,
        peak_bandwidth=args.peak_bandwidth,
        seqlen=args.seqlen,
        hidden_size=args.hidden_size,
        min_bs=args.min_bs,
        max_bs=args.max_bs,
        series_filter=args.series
    )
    
    if fig:
        fig.savefig(args.output, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved as {args.output}")

if __name__ == "__main__":
    main()