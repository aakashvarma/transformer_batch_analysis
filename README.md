# Multi-GPU Transformer Operation Benchmarking Tool

A comprehensive benchmarking tool for evaluating transformer operations across different NVIDIA GPUs (RTX 3090, A10, H100). This tool measures the performance of key transformer operations including dense matrix multiplication and attention mechanisms.

## Features

- Support for multiple NVIDIA GPU architectures:
  - NVIDIA RTX 3090 (24GB GDDR6X)
  - NVIDIA A10 (24GB GDDR6)
  - NVIDIA H100 (80GB HBM3)
- Benchmarks three key transformer operations:
  - Dense matrix multiplication
  - Query-Key attention initialization
  - Query-Key attention auto-regressive mode
- FP16 precision support
- Customizable memory limits
- Comprehensive CSV and pickle output formats
- Progress tracking with detailed metrics

## Prerequisites

```bash
# Required Python packages
pip install torch numpy pandas tqdm
```

System requirements:
- Python 3.7+
- CUDA-compatible GPU
- PyTorch with CUDA support
- Sufficient disk space for output files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transformer-benchmarks.git
cd transformer-benchmarks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run benchmarks on a specific GPU:
```bash
python benchmark.py --gpu a10     # For NVIDIA A10
python benchmark.py --gpu 3090    # For RTX 3090
python benchmark.py --gpu h100    # For H100
```

Run benchmarks on all supported GPUs:
```bash
python benchmark.py --all
```

### Advanced Options

Override default memory limits:
```bash
python benchmark.py --gpu a10 --custom-memory 24  # Set custom memory limit in GB
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--gpu` | Specify GPU type (choices: '3090', 'a10', 'h100') |
| `--all` | Run benchmarks on all available GPU types |
| `--custom-memory` | Override default GPU memory limit (in GB) |

## Output

The tool generates two types of output files in the `data/` directory:

1. CSV files (`transformer-batching-microbenchmarks-{gpu}-fp16-{date}.csv`):
   - Detailed metrics for each operation
   - Performance statistics
   - Hardware-specific information

2. Pickle files (`{date}-transformer-batching-{gpu}-fp16.pkl.gz`):
   - Raw benchmark data
   - Complete measurement results
   - Compressed format for efficient storage

### Output Metrics

The benchmark results include:
- Latency measurements
- Throughput calculations
- FLOP counts
- Memory I/O statistics
- Arithmetic intensity
- Batch size scaling
- Sequence length impact

## Benchmark Configurations

Default configurations tested:
- Model dimensions: Various combinations of n∈[12,16,32,40,56,72,96] and d∈[64,128]
- Sequence lengths: [10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 5000]
- Batch sizes: Range from 1 to 128 with variable increments

## Data Analysis

The output CSV files can be analyzed using standard data analysis tools:

```python
import pandas as pd

# Load benchmark results
results = pd.read_csv('data/transformer-batching-microbenchmarks-a10-fp16-20241124.csv')

# Basic analysis examples
throughput_stats = results.groupby('series')['throughput'].describe()
memory_efficiency = results.groupby(['series', 'bs'])['intensity'].mean()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License (see LICENSE file for details)

## Citation

If you use this benchmark tool in your research, please cite:

```bibtex
@software{transformer_benchmarks_2024,
  title={Multi-GPU Transformer Operation Benchmarking Tool},
  author={Aakash Varma},
  year={2024},
  url={https://github.com/yourusername/transformer-benchmarks}
}
```

## Troubleshooting

Common issues and solutions:

1. **Out of Memory Errors**
   - Reduce batch sizes or sequence lengths
   - Use `--custom-memory` to set appropriate memory limits

2. **CUDA Device Not Found**
   - Ensure CUDA toolkit is properly installed
   - Check GPU driver version compatibility
   - Verify PyTorch CUDA support with `torch.cuda.is_available()`

3. **Performance Issues**
   - Clear GPU cache between runs
   - Close other GPU-intensive applications
   - Monitor GPU temperature and throttling

## Support

For bug reports and feature requests, please use the GitHub issue tracker.

For questions and discussions, feel free to reach out to @varmology on X (previously Twitter).
