import struct
import matplotlib.pyplot as plt

class BenchmarkData:
    def __init__(
        self,
        element_counts: list[int],
        ms_values: list[float],
        bandwidth_values: list[float]):
        self.element_counts = element_counts
        self.ms_values = ms_values
        self.bandwidth_values = bandwidth_values

    element_counts: list[int]
    ms_values: list[float]
    bandwidth_values: list[float]

def load_benchmark_data(file_path: str) -> BenchmarkData:
    element_counts = []
    ms_values = []
    bandwidth_values = []
    with open(file_path, "rb") as file:
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
    return BenchmarkData(element_counts, ms_values, bandwidth_values)

thrust_data = load_benchmark_data("bench_reduce_thrust.bin")
reduce2_data = load_benchmark_data("bench_reduce2.bin")
reduce3_data = load_benchmark_data("bench_reduce3.bin")

plt.xlabel("Element Count")
plt.xscale('log', base=2)

plt.ylabel("Bandwidth (GB/s)")
plt.title("Reduce Bandwidth")

plt.plot(thrust_data.element_counts, thrust_data.bandwidth_values)
plt.plot(reduce2_data.element_counts, reduce2_data.bandwidth_values)
plt.plot(reduce3_data.element_counts, reduce3_data.bandwidth_values)
plt.legend(["Thrust", "Reduce2", "Reduce3"])
plt.savefig("bench_bandwidth.png")