import math
import struct
import sys
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    labels: tuple[str]
    gbps_values: tuple[float]
    gflops_values: tuple[float]
    ms_values: tuple[float]

def load_benchmark_result(path: str):
    labels = []
    gbps_values = []
    gflops_values = []
    ms_values = [] 
    with open(path, "rb") as f:
        while True:
            data = f.read(4)
            if not data:
                break

            label_length = struct.unpack("<I", data)[0]
            label = f.read(label_length).decode("utf-8")
            labels.append(label)

            gbps, gflops, ms = struct.unpack("<ddd", f.read(24))
            gbps_values.append(gbps)
            gflops_values.append(gflops)
            ms_values.append(ms)
    
    return BenchmarkResult(tuple(labels), tuple(gbps_values), tuple(gflops_values), tuple(ms_values))

def draw_bar_graph(labels, values, value_axis_label):
    bars = plt.bar(labels, values, edgecolor="black")
    
    plt.bar_label(bars)
    plt.xlabel("Matrix dimensions")
    plt.ylabel(value_axis_label)

    plt.xticks(rotation=45)

    plt.ylim(0, max(values)*1.1)
    plt.savefig("bench_matmul.png", bbox_inches='tight')

def draw_comparison_bar_graph(labels, value_axis_label, values_one, values_one_label, values_two, values_two_label):
    bar_width = 1.2
    label_pos_x = 4*np.arange(len(labels))

    bars_one = plt.bar(label_pos_x - bar_width/2, values_one, edgecolor="black", width=bar_width, label=values_one_label)
    bars_two = plt.bar(label_pos_x + bar_width/2, values_two, edgecolor="black", width=bar_width, label=values_two_label)
    plt.legend()
    
    plt.bar_label(bars_one, fontsize=8, bbox=dict(facecolor='white', alpha=0.4))
    plt.bar_label(bars_two, fontsize=8, bbox=dict(facecolor='white', alpha=0.4))

    plt.xlabel("Matrix dimensions")
    plt.ylabel(value_axis_label)

    plt.xticks(rotation=45)
    plt.xticks(label_pos_x, labels)

    plt.ylim(0, max(max(values_one), max(values_two))*1.1)
    plt.savefig(f"bench_matmul.png", bbox_inches='tight')

# if len(sys.argv) <= 1:
#     print("Usage: python plot_bench.py benchmark file(s)...")
#     sys.exit(1)
# 
# benchmark_files = sys.argv[1:]

# matmul_naive_results = load_benchmark_result("matmul_naive.bin")
# draw_bar_graph(matmul_naive_results.labels, matmul_naive_results.gflops_values, "GFLOPS/s")

matmul_naive_block_dim_16_results = load_benchmark_result("matmul_naive_block_dim_16.bin")
matmul_naive_block_dim_32_results = load_benchmark_result("matmul_naive_block_dim_32.bin")

draw_comparison_bar_graph(matmul_naive_block_dim_16_results.labels, "GFLOPS/s", matmul_naive_block_dim_16_results.gflops_values, "Block dim 16", matmul_naive_block_dim_32_results.gflops_values, "Block dim 32")

