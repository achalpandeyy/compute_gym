import struct
import matplotlib.pyplot as plt

class BenchmarkData:
    def __init__(
        self,
        peak_gbps: float,
        peak_gflops: float,
        element_counts: list[int],
        ms_values: list[float],
        bandwidth_values: list[float]):
        self.peak_gbps = peak_gbps
        self.peak_gflops = peak_gflops
        self.element_counts = element_counts
        self.ms_values = ms_values
        self.bandwidth_values = bandwidth_values

    peak_gbps: float
    peak_gflops: float
    element_counts: list[int]
    ms_values: list[float]
    bandwidth_values: list[float]

def load_benchmark_data(file_path: str) -> BenchmarkData:
    element_counts = []
    ms_values = []
    bandwidth_values = []
    with open(file_path, "rb") as file:
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

thrust_data = load_benchmark_data("bench_reduce_thrust.bin")
reduce2_data = load_benchmark_data("bench_reduce2.bin")
reduce3_data = load_benchmark_data("bench_reduce3.bin")
reduce4_data = load_benchmark_data("bench_reduce4.bin")
reduce5_data = load_benchmark_data("bench_reduce5.bin")

assert reduce2_data.peak_gbps == reduce3_data.peak_gbps == reduce4_data.peak_gbps == reduce5_data.peak_gbps == thrust_data.peak_gbps
assert reduce2_data.peak_gflops == reduce3_data.peak_gflops == reduce4_data.peak_gflops == reduce5_data.peak_gflops == thrust_data.peak_gflops
assert reduce2_data.element_counts == reduce3_data.element_counts == reduce4_data.element_counts == reduce5_data.element_counts == thrust_data.element_counts

# Draw plot

plt.xlabel("Element Count")
plt.xscale('log', base=2)

plt.ylabel("Bandwidth (GBPS)")
plt.title("Reduce Bandwidth")

plt.hlines(reduce3_data.peak_gbps, reduce3_data.element_counts[0], reduce3_data.element_counts[-1], colors="red", linestyles="dashed")
plt.plot(thrust_data.element_counts, thrust_data.bandwidth_values)
plt.plot(reduce2_data.element_counts, reduce2_data.bandwidth_values)
plt.plot(reduce3_data.element_counts, reduce3_data.bandwidth_values)
plt.plot(reduce4_data.element_counts, reduce4_data.bandwidth_values)
plt.plot(reduce5_data.element_counts, reduce5_data.bandwidth_values)
# plt.plot(copy_data.element_counts, copy_data.bandwidth_values)
plt.legend(["Peak", "Thrust", "Reduce2", "Reduce3", "Reduce4", "Reduce5"], loc="lower right")
plt.savefig("bench_bandwidth.png")