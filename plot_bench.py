import math
import struct
import sys
import matplotlib.pyplot as plt

class BenchmarkData:
    def __init__(
        self,
        file_name: str,
        peak_gbps: float | None,
        peak_gflops: float | None,
        element_counts: list[int],
        ms_values: list[float],
        bandwidth_values: list[float]):
        self.file_name = file_name
        self.peak_gbps = peak_gbps
        self.peak_gflops = peak_gflops
        self.element_counts = element_counts
        self.ms_values = ms_values
        self.bandwidth_values = bandwidth_values

    file_name: str
    peak_gbps: float | None
    peak_gflops: float | None
    element_counts: list[int]
    ms_values: list[float]
    bandwidth_values: list[float]

def load_benchmark_data(file_name: str) -> BenchmarkData:
    element_counts = []
    ms_values = []
    bandwidth_values = []
    with open(file_name, "rb") as file:
        peak_gbps = None
        peak_gflops = None
        metadata_present = struct.unpack("b", file.read(1))[0]
        if metadata_present:
            peak_gbps = struct.unpack("d", file.read(8))[0]
            peak_gflops = struct.unpack("d", file.read(8))[0]

        record_format = "Qdd"
        record_size = struct.calcsize(record_format)

        while True:
            chunk = file.read(record_size)
            if not chunk:
                break
            element_counts.append(struct.unpack(record_format, chunk)[0])
            ms_values.append(struct.unpack(record_format, chunk)[1])
            bandwidth_values.append(struct.unpack(record_format, chunk)[2])

    assert len(element_counts) == len(ms_values) == len(bandwidth_values)
    return BenchmarkData(file_name, peak_gbps, peak_gflops, element_counts, ms_values, bandwidth_values)

if len(sys.argv) <= 1:
    print("Usage: python plot_bench.py benchmark file(s)...")
    sys.exit(1)

benchmark_files = sys.argv[1:]

benchmark_datas = []
element_counts: list[int] | None = None
peak_gbps: float | None = None
peak_gflops: float | None = None
for benchmark_file in benchmark_files:
    benchmark_datas.append(load_benchmark_data(benchmark_file))

    if element_counts:  
        assert element_counts == benchmark_datas[-1].element_counts
    else:
        element_counts = benchmark_datas[-1].element_counts
    if peak_gbps:
        assert peak_gbps == benchmark_datas[-1].peak_gbps
    else:
        peak_gbps = benchmark_datas[-1].peak_gbps
    if peak_gflops:
        assert peak_gflops == benchmark_datas[-1].peak_gflops
    else:
        peak_gflops = benchmark_datas[-1].peak_gflops

assert element_counts
assert peak_gbps
assert peak_gflops

# Draw plot
plt.title("Bandwidth")
fig, ax = plt.subplots(figsize=(15, 8))

x_range = math.log2(element_counts[-1]) - math.log2(element_counts[0])
ax.set_xlabel("Element Count")
ax.set_xscale('log', base=2)
plt.xlim(element_counts[0], element_counts[-1])
ax.set_xticks(element_counts)

ax.set_ylabel("Bandwidth (GBPS)")
# ax.set_aspect(15/450, adjustable='box')
start_y = 0
end_y = 450 # TODO(achal): This should be calculated based on the X (visual) tick size 
h = (end_y - start_y) / len(element_counts)
ax.set_yticks([start_y + i*h for i in range(len(element_counts))])

# Add grid and set background
plt.grid(True, alpha=0.3)  # Add grid with transparency
plt.gca().set_facecolor('black')  # Set plot background to black

if peak_gbps:
    plt.hlines(peak_gbps, element_counts[0], element_counts[-1], colors="red", linestyles="dashed", linewidth=1)

legend_entries = []
for benchmark_data in benchmark_datas:
    plt.plot(benchmark_data.element_counts, benchmark_data.bandwidth_values, linewidth=2)
    legend_entries.append(benchmark_data.file_name)

plt.legend([
    "Peak",
    *legend_entries,
], 
    bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("bench_bandwidth.png", bbox_inches='tight')