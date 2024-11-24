#!/usr/bin/env python3

"""
Transformer Operation Benchmarking Script for A10 GPU

This script benchmarks various transformer operations on A10 GPU including:
1. Dense matrix multiplication
2. Query-Key attention (initialization phase)
3. Query-Key attention (auto-regressive phase)

Optimized for A10's features:
- ~24GB GDDR6 memory
- Appropriate memory bandwidth
- FP16 precision support
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
import pickle
import gzip
from datetime import datetime
from tqdm.auto import tqdm
import torch

# Disable gradient computation for benchmarking
torch.set_grad_enabled(False)

# Benchmark configurations
ND_LIST = list(itertools.chain(itertools.product([12, 16, 32], [64]), itertools.product([32, 40, 56, 72, 96], [128])))
SEQLEN_LIST = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 5000]
BS_LIST = list(itertools.chain(range(1, 8), range(8, 16, 2), range(16, 32, 4), range(32, 64, 8), range(64, 128, 16), [128]))

def benchmark(f, *, f_setup=None, min_repeat: int, min_secs: float, tqdm_kwargs: dict | None=None) -> np.ndarray:
    """
    Benchmark a function by running it multiple times and measuring latency.
    """
    latency = []
    
    # First run, ignore min_secs
    if f_setup is not None:
        f_setup()
    st = time.perf_counter_ns()
    f()
    ed = time.perf_counter_ns()
    latency.append((ed-st)/1e9)
    
    # Subsequent runs, until reaching both min_repeat and min_secs
    min_nanos = int(min_secs * 1e9)
    start_nanos = time.perf_counter_ns()
    while True:
        now_nanos = time.perf_counter_ns()
        if len(latency) > min_repeat and now_nanos - start_nanos > min_nanos:
            break
        if f_setup is not None:
            f_setup()
        st = time.perf_counter_ns()
        f()
        ed = time.perf_counter_ns()
        latency.append((ed-st)/1e9)
    return np.array(latency)

def tail_mean(xs, skip=0.2):
    """Calculate mean of array after skipping initial portion."""
    return xs[int(len(xs) * skip):].mean()

def benchmark_dense(out, nd_list, seqlen_list, bs_list):
    """
    Benchmark dense matrix multiplication operations using FP16.
    """
    seqlen_list = [1] + seqlen_list
    total = len(list(itertools.product(nd_list, seqlen_list, bs_list)))
    pbar = tqdm(total=total)
    
    for (n, d), seqlen in reversed(list(itertools.product(nd_list, seqlen_list))):
        h = n * d
        try:
            maxbs = max(b for b in bs_list if b*seqlen*h*2 + h*h*2 + b*seqlen*h*2 < 24e9)
        except ValueError:
            pbar.update(len(bs_list))
            continue
            
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda:0")
        
        X = torch.rand((maxbs, seqlen, h), dtype=torch.float16, device="cuda:0")
        W = torch.rand((h, h), dtype=torch.float16, device="cuda:0")
            
        torch.cuda.synchronize()
        for bs in reversed(bs_list):
            if bs > maxbs:
                pbar.update()
                continue
                
            pbar.set_postfix(n=n, h=h, d=d, seqlen=seqlen, bs=bs)
            def run():
                torch.matmul(X[:bs], W)
                torch.cuda.synchronize()
                
            def clear_cache():
                cache.zero_()
                torch.cuda.synchronize()
                
            latency = benchmark(run, f_setup=clear_cache, min_repeat=20, min_secs=2)
            l = tail_mean(latency)
            out.append({
                "n": n,
                "d": d,
                "seqlen": seqlen,
                "bs": bs,
                "latency": l
            })
            pbar.update()
        del cache, X, W
        torch.cuda.empty_cache()
    pbar.close()

def benchmark_qk_init(out, nd_list, seqlen_list, bs_list):
    """
    Benchmark Query-Key attention initialization.
    Optimized for A10's memory capacity.
    """
    total = len(list(itertools.product(nd_list, seqlen_list, bs_list)))
    pbar = tqdm(total=total)
    for (n, d), seqlen in reversed(list(itertools.product(nd_list, seqlen_list))):
        h = n * d
        try:
            # Adjusted memory limit for A10 (~24GB)
            maxbs = max(b for b in bs_list if b*n*seqlen*d*2*2+b*n*seqlen**2*2 < 24e9)
        except ValueError:
            pbar.update(len(bs_list))
            continue
            
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda:0")
        Qmax = torch.rand((maxbs, n, seqlen, d), dtype=torch.float16, device="cuda:0")
        Kmax = torch.rand((maxbs, n, seqlen, d), dtype=torch.float16, device="cuda:0")
        torch.cuda.synchronize()
        for bs in reversed(bs_list):
            pbar.set_postfix(n=n, h=h, d=d, seqlen=seqlen, bs=bs)
            if bs > maxbs:
                pbar.update()
                continue
            Q = Qmax[:bs]
            K = Kmax[:bs]
            def run():
                torch.bmm(Q.view(bs * n, seqlen, d), 
                         K.view(bs * n, seqlen, d).transpose(1, 2))
                torch.cuda.synchronize()
            def clear_cache():
                cache.zero_()
                torch.cuda.synchronize()
            latency = benchmark(run, f_setup=clear_cache, min_repeat=20, min_secs=2)
            l = tail_mean(latency)
            out.append({
                "n": n,
                "d": d,
                "seqlen": seqlen,
                "bs": bs,
                "latency": l
            })
            pbar.update()
        del cache, Q, K, Qmax, Kmax
        torch.cuda.empty_cache()
    pbar.close()

def benchmark_qk_ar(out, nd_list, seqlen_list, bs_list):
    """
    Benchmark Query-Key attention in auto-regressive mode.
    Optimized for A10 capabilities.
    """
    total = len(list(itertools.product(nd_list, seqlen_list, bs_list)))
    pbar = tqdm(total=total)
    for (n, d), seqlen in reversed(list(itertools.product(nd_list, seqlen_list))):
        h = n * d
        try:
            # Adjusted memory limit for A10
            maxbs = max(b for b in bs_list if b*n*(1+seqlen)*d*2+b*n*seqlen*2 < 24e9)
        except ValueError:
            pbar.update(len(bs_list))
            continue
            
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda:0")
        
        # Use FP16 for all computations
        Qmax = torch.rand((maxbs, n, 1, d), dtype=torch.float16, device="cuda:0")
        Kmax = torch.rand((maxbs, n, seqlen, d), dtype=torch.float16, device="cuda:0")
            
        torch.cuda.synchronize()
        
        for bs in reversed(bs_list):
            pbar.set_postfix(n=n, h=h, d=d, seqlen=seqlen, bs=bs)
            if bs > maxbs:
                pbar.update()
                continue
                
            Q = Qmax[:bs]
            K = Kmax[:bs]
            
            def run():
                torch.bmm(Q.view(bs * n, 1, d), 
                         K.view(bs * n, seqlen, d).transpose(1, 2))
                torch.cuda.synchronize()
                
            def clear_cache():
                cache.zero_()
                torch.cuda.synchronize()
                
            latency = benchmark(run, f_setup=clear_cache, min_repeat=20, min_secs=2)
            l = tail_mean(latency)
            out.append({
                "n": n,
                "d": d,
                "seqlen": seqlen,
                "bs": bs,
                "latency": l
            })
            pbar.update()
        del cache, Q, K, Qmax, Kmax
        torch.cuda.empty_cache()
    pbar.close()

def process_results(data):
    """Process benchmark results and save as CSV."""
    df_dense = (
        pd.DataFrame.from_dict(data["dense"])
        .assign(h=lambda x: x["n"] * x["d"])
        .assign(flop=lambda x: (x["bs"] * x["seqlen"] * x["h"]**2) * 2)
        .assign(io=lambda x: (x["bs"]*x["seqlen"]*x["h"]*2 + x["h"]**2) * 2)
        .assign(intensity=lambda x: x["flop"] / x["io"])
        .assign(throughput=lambda x: x["bs"]*x["seqlen"] / x["latency"])
        .assign(series="dense")
    )
    df_qk_init = (
        pd.DataFrame.from_dict(data["qk_init"])
        .assign(h=lambda x: x["n"] * x["d"])
        .assign(flop=lambda x: (x["bs"]*x["n"]*x["d"]*x["seqlen"]**2) * 2)
        .assign(io=lambda x: (x["bs"]*x["n"]*(x["seqlen"]*x["d"]*2 + x["seqlen"]**2)) * 2)
        .assign(intensity=lambda x: x["flop"] / x["io"])
        .assign(throughput=lambda x: x["bs"]*x["seqlen"] / x["latency"])
        .assign(series="qk_init")
    )
    df_qk_ar = (
        pd.DataFrame.from_dict(data["qk_ar"])
        .assign(h=lambda x: x["n"] * x["d"])
        .assign(flop=lambda x: (x["bs"]*x["n"]*x["d"]*x["seqlen"]) * 2)
        .assign(io=lambda x: (x["bs"]*x["n"]*(x["d"] + x["seqlen"]*x["d"] + x["seqlen"])) * 2)
        .assign(intensity=lambda x: x["flop"] / x["io"])
        .assign(throughput=lambda x: x["bs"] / x["latency"])
        .assign(series="qk_ar")
    )
    
    # Add A10-specific metrics
    for df in [df_dense, df_qk_init, df_qk_ar]:
        df['gpu'] = 'A10'
        df['precision'] = 'fp16'

    # Combine and save all results
    timestamp = datetime.now().strftime("%Y%m%d")
    pd.concat([df_dense, df_qk_init, df_qk_ar]).to_csv(
        f"data/transformer-batching-microbenchmarks-a10-fp16-{timestamp}.csv", 
        index=False
    )

def main():
    # Run benchmarks
    data = {}
    
    print("Running Query-Key initialization benchmarks...")
    db = []
    benchmark_qk_init(db, ND_LIST, SEQLEN_LIST, BS_LIST)
    data["qk_init"] = db

    print("Running Query-Key auto-regressive benchmarks...")
    db = []
    benchmark_qk_ar(db, ND_LIST, SEQLEN_LIST, BS_LIST)
    data["qk_ar"] = db

    print("Running dense operation benchmarks...")
    db = []
    benchmark_dense(db, ND_LIST, SEQLEN_LIST, BS_LIST)
    data["dense"] = db

    # Save benchmark results
    timestamp = datetime.now().strftime("%Y%m%d")
    with gzip.open(f"data/{timestamp}-transformer-batching-a10-fp16.pkl.gz", "wb") as f:
        pickle.dump(data, f)

    # Process and save results as CSV
    process_results(data)

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Print configuration and A10-specific information
    print("Starting transformer benchmarking on A10 GPU...")
    print("\nA10 Specific Features:")
    print("- Using FP16 precision for all operations")
    print("- Memory capacity: ~24GB GDDR6")
    print("- Optimized for A10 memory bandwidth")
    
    print("\nBenchmarking configurations:")
    print("- Model configurations:", ND_LIST)
    print("- Sequence lengths:", SEQLEN_LIST)
    print("- Batch sizes:", BS_LIST)
    
    # Run main benchmark suite
    main()