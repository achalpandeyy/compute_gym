import math
import struct
import matplotlib.pyplot as plt

class BenchmarkData:
    def __init__(
        self,
        peak_gbps: float | None,
        peak_gflops: float | None,
        element_counts: list[int],
        ms_values: list[float],
        bandwidth_values: list[float]):
        self.peak_gbps = peak_gbps
        self.peak_gflops = peak_gflops
        self.element_counts = element_counts
        self.ms_values = ms_values
        self.bandwidth_values = bandwidth_values

    peak_gbps: float | None
    peak_gflops: float | None
    element_counts: list[int]
    ms_values: list[float]
    bandwidth_values: list[float]

def load_benchmark_data(file_path: str) -> BenchmarkData:
    element_counts = []
    ms_values = []
    bandwidth_values = []
    with open(file_path, "rb") as file:
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
    return BenchmarkData(peak_gbps, peak_gflops, element_counts, ms_values, bandwidth_values)

reduce2_data = load_benchmark_data("bench_reduce2.bin")
reduce3_data = load_benchmark_data("bench_reduce3.bin")
reduce4_data = load_benchmark_data("bench_reduce4.bin")
reduce5_data = load_benchmark_data("bench_reduce5.bin")
thrust_data = load_benchmark_data("bench_reduce_thrust.bin")

# vectorsum5_data = load_benchmark_data("bench_vectorsum_5.bin")
# vectorsum_triton_data = load_benchmark_data("bench_vectorsum_triton.bin")

# assert reduce2_data.peak_gbps == reduce3_data.peak_gbps == reduce4_data.peak_gbps == reduce5_data.peak_gbps == thrust_data.peak_gbps
# assert reduce2_data.peak_gflops == reduce3_data.peak_gflops == reduce4_data.peak_gflops == reduce5_data.peak_gflops == thrust_data.peak_gflops
# assert reduce2_data.element_counts == reduce3_data.element_counts == reduce4_data.element_counts == reduce5_data.element_counts == thrust_data.element_counts
element_counts = reduce2_data.element_counts
bandwidth_values = reduce2_data.bandwidth_values

# Draw plot
plt.title("Reduce Bandwidth")
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

if reduce2_data.peak_gbps is not None:
    plt.hlines(reduce2_data.peak_gbps, reduce2_data.element_counts[0], reduce2_data.element_counts[-1], colors="red", linestyles="dashed", linewidth=1)
breakpoint()
plt.plot(thrust_data.element_counts, thrust_data.bandwidth_values, color='green', linewidth=2)
# plt.plot(reduce2_data.element_counts, reduce2_data.bandwidth_values, color='cyan', linewidth=1)
# plt.plot(reduce3_data.element_counts, reduce3_data.bandwidth_values, color='cyan', linewidth=1)
# plt.plot(reduce4_data.element_counts, reduce4_data.bandwidth_values, color='cyan', linewidth=1)
plt.plot(reduce5_data.element_counts, reduce5_data.bandwidth_values, color='cyan', linewidth=1)
# plt.plot(vectorsum5_data.element_counts, vectorsum5_data.bandwidth_values, color='white', linewidth=1)
# plt.plot(vectorsum_triton_data.element_counts, vectorsum_triton_data.bandwidth_values, color='blue', linewidth=1)

plt.legend([
    "Peak",
    "Thrust",
    # "Reduce2",
    # "Reduce3",
    # "Reduce4",
    "Reduce5",
    "vectorsum5",
    "vectorsum_triton"], 
    bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("bench_bandwidth.png", bbox_inches='tight')