"""
Performance Analysis Plotting Tool
--------------------------------

This module provides functionality for analyzing and visualizing performance metrics
from machine learning model benchmarks. It focuses on analyzing relationships between
various parameters like sequence length, hidden size, batch size and their impact on
model performance metrics such as throughput and latency.

Key Features:
- Throughput vs FLOPs analysis
- Latency vs sequence length analysis
- Arithmetic intensity analysis
- Batch size impact analysis
- Support for different model series (dense_init, dense_ar, qk_init, qk_ar)
- Support for multiple dtypes
- Configurable filtering options
- Comprehensive plotting capabilities

Usage:
    Basic:
    python script.py data.csv --output_dir ./plots

    With filters:
    python script.py data.csv --output_dir ./plots --min_seqlen 2 --max_seqlen 1000 
                             --min_h 1024 --max_h 4096 --min_bs 1 --max_bs 128

Requirements:
    - pandas
    - matplotlib
    - seaborn
    - numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Constants
STYLE_CONFIG = {
    'style': 'default',
    'font_family': 'monospace',
    'label_size': 10,
    'title_size': 12,
    'tick_size': 9,
    'grid': True,
    'grid_alpha': 0.1,
    'face_color': 'white'
}

SERIES_ORDER = ['dense_init', 'dense_ar', 'qk_init', 'qk_ar']
COLOR_SCHEME = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

def setup_matplotlib_style() -> None:
    """Configure matplotlib plotting style parameters for consistent visualization."""
    plt.style.use(STYLE_CONFIG['style'])
    plt.rcParams.update({
        'font.family': STYLE_CONFIG['font_family'],
        'axes.labelsize': STYLE_CONFIG['label_size'],
        'axes.titlesize': STYLE_CONFIG['title_size'],
        'xtick.labelsize': STYLE_CONFIG['tick_size'],
        'ytick.labelsize': STYLE_CONFIG['tick_size'],
        'axes.grid': STYLE_CONFIG['grid'],
        'grid.alpha': STYLE_CONFIG['grid_alpha'],
        'axes.facecolor': STYLE_CONFIG['face_color'],
        'figure.facecolor': STYLE_CONFIG['face_color']
    })

class PerformanceDataLoader:
    """Handles loading and preprocessing of performance data from CSV files."""
    
    @staticmethod
    def load_and_process_data(
        csv_path: str,
        min_h: Optional[int] = None,
        max_h: Optional[int] = None,
        min_bs: Optional[int] = None,
        max_bs: Optional[int] = None,
        min_seqlen: Optional[int] = None,
        max_seqlen: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load and process the CSV file with optional filtering parameters.
        
        Args:
            csv_path: Path to the CSV file containing performance data
            min_h: Minimum hidden size to include
            max_h: Maximum hidden size to include
            min_bs: Minimum batch size to include
            max_bs: Maximum batch size to include
            min_seqlen: Minimum sequence length to include
            max_seqlen: Maximum sequence length to include
            
        Returns:
            Processed DataFrame with applied filters and renamed series
        """
        df = pd.read_csv(csv_path)
        
        # Process series names
        dense_init_mask = (df['series'] == 'dense') & (df['seqlen'] == 1)
        df.loc[dense_init_mask, 'series'] = 'dense_init'
        df.loc[df['series'] == 'dense', 'series'] = 'dense_ar'
        
        # Apply filters
        filters = {
            'h': (min_h, max_h),
            'bs': (min_bs, max_bs),
            'seqlen': (min_seqlen, max_seqlen)
        }
        
        for column, (min_val, max_val) in filters.items():
            if min_val is not None:
                df = df[df[column] >= min_val]
            if max_val is not None:
                df = df[df[column] <= max_val]
        
        return df

class PerformancePlotter:
    """Handles creation of various performance analysis plots."""
    
    def __init__(self, output_dir: str):
        """Initialize with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        setup_matplotlib_style()
    
    def plot_throughput_vs_flops(self, df: pd.DataFrame, min_seqlen: int) -> None:
        """Generate throughput vs FLOPs plots for each series and dtype."""
        for series in SERIES_ORDER:
            for dtype in df['dtype'].unique():
                fig, ax = plt.subplots(figsize=(12, 8))
                data = df[(df['dtype'] == dtype) & (df['series'] == series)]
                
                colors = plt.cm.rainbow(np.linspace(0, 1, len(data['h'].unique())))
                for h_val, color in zip(sorted(data['h'].unique()), colors):
                    subset = data[data['h'] == h_val]
                    plt.scatter(subset['flop'], subset['throughput'],
                              label=f'h={h_val}', alpha=0.6, color=color)
                
                self._format_plot(ax, 'FLOPs', 'Throughput (Tokens/s)',
                                f'{series.upper()} Throughput vs FLOPs for {dtype}')
                plt.savefig(os.path.join(self.output_dir, f'{series}_throughput_vs_flops_{dtype}.png'),
                           bbox_inches='tight', dpi=300, facecolor='white')
                plt.close()
    
    def plot_latency_vs_batchsize(self, df: pd.DataFrame) -> None:
        """Generate latency vs batch size plots."""
        # Plot for dense series
        for series in ['dense_init', 'dense_ar']:
            for dtype in df['dtype'].unique():
                self._plot_dense_latency(df, series, dtype)
        
        # Plot for QK_INIT (grouped by sequence length)
        for dtype in df['dtype'].unique():
            self._plot_qk_init_latency(df, dtype)
        
        # Plot for QK_AR (grouped by hidden size)
        for dtype in df['dtype'].unique():
            self._plot_qk_ar_latency(df, dtype)
    
    def _plot_dense_latency(self, df: pd.DataFrame, series: str, dtype: str) -> None:
        """Plot latency analysis for dense series."""
        data = df[(df['dtype'] == dtype) & (df['series'] == series)]
        if len(data) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        h_values = sorted(data['h'].unique())
        colors = COLOR_SCHEME[:len(h_values)]
        
        for h_val, color in zip(h_values, colors):
            subset = data[data['h'] == h_val].sort_values('bs')
            plt.scatter(subset['bs'], subset['latency'], color=color, alpha=0.3, s=20)
            
            mean_latency = subset.groupby('bs')['latency'].mean().reset_index()
            plt.plot(mean_latency['bs'], mean_latency['latency'],
                    '-', color=color, label=f'h={h_val}', linewidth=2)
        
        self._format_plot(ax, 'Batch Size', 'Latency (ms)',
                         f'{series.upper()} Latency vs Batch Size\nGrouped by Hidden Size (h) - {dtype}')
        plt.savefig(os.path.join(self.output_dir, f'{series}_latency_vs_bs_by_h_{dtype}.png'),
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
    
    def _plot_qk_init_latency(self, df: pd.DataFrame, dtype: str) -> None:
        """Plot latency analysis for QK_INIT."""
        data = df[(df['dtype'] == dtype) & (df['series'] == 'qk_init')]
        if len(data) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        seqlens = sorted(data['seqlen'].unique())
        colors = COLOR_SCHEME[:len(seqlens)]
        
        for seqlen, color in zip(seqlens, colors):
            subset = data[data['seqlen'] == seqlen].sort_values('bs')
            plt.scatter(subset['bs'], subset['latency'], color=color, alpha=0.3, s=20)
            
            mean_latency = subset.groupby('bs')['latency'].mean().reset_index()
            plt.plot(mean_latency['bs'], mean_latency['latency'],
                    '-', color=color, label=f'seqlen={seqlen}', linewidth=2)
        
        self._format_plot(ax, 'Batch Size', 'Latency (ms)',
                         f'QK_INIT Latency vs Batch Size\nGrouped by Sequence Length - {dtype}')
        plt.savefig(os.path.join(self.output_dir, f'qk_init_latency_vs_bs_by_seqlen_{dtype}.png'),
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
    
    def _plot_qk_ar_latency(self, df: pd.DataFrame, dtype: str) -> None:
        """Plot latency analysis for QK_AR."""
        data = df[(df['dtype'] == dtype) & (df['series'] == 'qk_ar')]
        if len(data) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        h_values = sorted(data['h'].unique())
        colors = COLOR_SCHEME[:len(h_values)]
        
        for h_val, color in zip(h_values, colors):
            subset = data[data['h'] == h_val].sort_values('bs')
            plt.scatter(subset['bs'], subset['latency'], color=color, alpha=0.3, s=20)
            
            mean_latency = subset.groupby('bs')['latency'].mean().reset_index()
            plt.plot(mean_latency['bs'], mean_latency['latency'],
                    '-', color=color, label=f'h={h_val}', linewidth=2)
        
        self._format_plot(ax, 'Batch Size', 'Latency (ms)',
                         f'QK_AR Latency vs Batch Size\nGrouped by Hidden Size (h) - {dtype}')
        plt.savefig(os.path.join(self.output_dir, f'qk_ar_latency_vs_bs_by_h_{dtype}.png'),
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
    
    def plot_arithmetic_intensity(self, df: pd.DataFrame, min_seqlen: int) -> None:
        """Generate arithmetic intensity analysis plots."""
        for series in SERIES_ORDER:
            for dtype in df['dtype'].unique():
                fig, ax = plt.subplots(figsize=(12, 8))
                data = df[(df['dtype'] == dtype) & (df['series'] == series)]
                
                colors = plt.cm.rainbow(np.linspace(0, 1, len(data['h'].unique())))
                for h_val, color in zip(sorted(data['h'].unique()), colors):
                    subset = data[data['h'] == h_val]
                    plt.scatter(subset['intensity'], subset['throughput'],
                              label=f'h={h_val}', alpha=0.6, color=color)
                
                self._format_plot(ax, 'Arithmetic Intensity (FLOPs/Byte)', 'Throughput (Tokens/s)',
                                f'{series.upper()} Roofline Analysis: Throughput vs Arithmetic Intensity for {dtype}')
                plt.savefig(os.path.join(self.output_dir, f'{series}_arithmetic_intensity_{dtype}.png'),
                           bbox_inches='tight', dpi=300, facecolor='white')
                plt.close()
    
    def plot_batch_size_impact(self, df: pd.DataFrame, min_seqlen: int) -> None:
        """Generate batch size impact plots."""
        for series in SERIES_ORDER:
            for dtype in df['dtype'].unique():
                data = df[(df['dtype'] == dtype) & (df['series'] == series)]
                if len(data) == 0:
                    continue
                
                fig, ax = plt.subplots(figsize=(12, 8))
                seqlens = sorted(data['seqlen'].unique())
                h_values = sorted(data['h'].unique())
                colors = plt.cm.rainbow(np.linspace(0, 1, len(h_values)))
                
                legend_elements = []
                for h_val, color in zip(h_values, colors):
                    h_data = data[data['h'] == h_val]
                    
                    for seqlen in seqlens:
                        subset = h_data[h_data['seqlen'] == seqlen]
                        if len(subset) > 0:
                            subset_sorted = subset.sort_values('bs')
                            plt.scatter(subset_sorted['bs'], subset_sorted['throughput'],
                                      color=color, alpha=0.3, s=20)
                            
                            mean_throughput = subset.groupby('bs')['throughput'].mean().reset_index()
                            line = plt.plot(mean_throughput['bs'], mean_throughput['throughput'],
                                          '-', color=color, linewidth=2)
                            
                            if seqlen == seqlens[0]:
                                legend_elements.append(line[0])
                
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('Batch Size', fontsize=11, fontweight='bold', labelpad=10)
                plt.ylabel('Throughput (Tokens/s)', fontsize=11, fontweight='bold', labelpad=10)
                plt.title(f'{series.upper()} Throughput vs Batch Size for {dtype}',
                         pad=20, fontweight='bold')
                
                plt.legend(legend_elements,
                          [f'h={h}' for h in h_values],
                          loc='center left',
                          bbox_to_anchor=(1.02, 0.5),
                          fontsize=9)
                
                plt.text(0.98, 0.02,
                        f'Sequence lengths: {", ".join(map(str, seqlens))}',
                        transform=ax.transAxes,
                        ha='right',
                        va='bottom',
                        fontsize=8,
                        alpha=0.7)
                
                ax.grid(True, which="major", ls="-", alpha=0.2, color='#E5E5E5')
                ax.grid(True, which="minor", ls="-", alpha=0.1, color='#E5E5E5')
                ax.set_axisbelow(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{series}_batch_size_impact_{dtype}.png'),
                           bbox_inches='tight', dpi=300, facecolor='white')
                plt.close()
    
    @staticmethod
    def _format_plot(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
        """Apply consistent formatting to plot axes."""
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(xlabel, fontsize=11, fontweight='bold', labelpad=10)
        plt.ylabel(ylabel, fontsize=11, fontweight='bold', labelpad=10)
        plt.title(title, pad=20, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        ax.grid(True, which="major", ls="-", alpha=0.2, color='#E5E5E5')
        ax.grid(True, which="minor", ls="-", alpha=0.1, color='#E5E5E5')
        ax.set_axisbelow(True)
        plt.tight_layout()

class PerformanceAnalyzer:
    """Main class for running performance analysis and generating visualizations."""
    
    def __init__(self):
        """Initialize the analyzer with command line arguments."""
        self.args = self._parse_arguments()
        self.plotter = PerformancePlotter(self.args.output_dir)
    
    @staticmethod
    def _parse_arguments() -> argparse.Namespace:
        """Parse and return command line arguments."""
        parser = argparse.ArgumentParser(description='Generate performance analysis plots')
        parser.add_argument('csv_path', type=str, help='Path to the CSV file')
        parser.add_argument('--output_dir', type=str, default='.',
                          help='Directory to save the plots (default: current directory)')
        parser.add_argument('--min_seqlen', type=int, default=1,
                          help='Minimum sequence length to include (default: 1)')
        parser.add_argument('--max_seqlen', type=int, default=None,
                          help='Maximum sequence length to include (default: None)')
        parser.add_argument('--min_h', type=int, default=None,
                          help='Minimum hidden size to include (default: None)')
        parser.add_argument('--max_h', type=int, default=None,
                          help='Maximum hidden size to include (default: None)')
        parser.add_argument('--min_bs', type=int, default=None,
                          help='Minimum batch size to include (default: None)')
        parser.add_argument('--max_bs', type=int, default=None,
                          help='Maximum batch size to include (default: None)')
        return parser.parse_args()
    
    def run_analysis(self) -> None:
        """Execute the complete performance analysis workflow."""
        self._print_configuration()
        
        # Load and process data
        df = PerformanceDataLoader.load_and_process_data(
            self.args.csv_path,
            min_h=self.args.min_h,
            max_h=self.args.max_h,
            min_bs=self.args.min_bs,
            max_bs=self.args.max_bs,
            min_seqlen=self.args.min_seqlen,
            max_seqlen=self.args.max_seqlen
        )
        
        # Generate plots
        print("Generating plots...")
        self.plotter.plot_throughput_vs_flops(df, self.args.min_seqlen)
        self.plotter.plot_latency_vs_batchsize(df)
        self.plotter.plot_arithmetic_intensity(df, self.args.min_seqlen)
        self.plotter.plot_batch_size_impact(df, self.args.min_seqlen)
        
        # Print summary statistics
        self._print_summary_statistics(df)
    
    def _print_configuration(self) -> None:
        """Print the current configuration settings."""
        print(f"Loading data from {self.args.csv_path}")
        print("Applying filters:")
        if self.args.min_seqlen > 1:
            print(f"  - Minimum sequence length: {self.args.min_seqlen}")
        if self.args.max_seqlen:
            print(f"  - Maximum sequence length: {self.args.max_seqlen}")
        if self.args.min_h:
            print(f"  - Minimum hidden size: {self.args.min_h}")
        if self.args.max_h:
            print(f"  - Maximum hidden size: {self.args.max_h}")
        if self.args.min_bs:
            print(f"  - Minimum batch size: {self.args.min_bs}")
        if self.args.max_bs:
            print(f"  - Maximum batch size: {self.args.max_bs}")
    
    def _print_summary_statistics(self, df: pd.DataFrame) -> None:
        """Print summary statistics for the analyzed data."""
        print("\nSummary Statistics by dtype and series:")
        for dtype in df['dtype'].unique():
            print(f"\nDtype: {dtype}")
            for series in SERIES_ORDER:
                subset = df[(df['dtype'] == dtype) & (df['series'] == series)]
                if len(subset) > 0:
                    print(f"\n{series.upper()}:")
                    print(f"Average Throughput: {subset['throughput'].mean():,.2f} Tokens/s")
                    print(f"Average Latency: {subset['latency'].mean():,.2f} ms")
                    print(f"Maximum Throughput: {subset['throughput'].max():,.2f} Tokens/s")
                    print(f"Minimum Latency: {subset['latency'].min():,.2f} ms")
        
        print(f"\nPlots have been saved to {self.args.output_dir}")

def main():
    """Main entry point for the performance analysis tool."""
    analyzer = PerformanceAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()