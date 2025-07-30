# Method,Duration_ms,Throughput_GOPS,Correct,Matrix_Size
# Serial,0.236342,1.10917,1,4096
# SIMD,0.0310334,8.44716,1,4096
# GPU,1.50503,0.174178,1,4096
# Serial,1.92898,1.08718,1,16384
# SIMD,0.184575,11.3621,1,16384
# GPU,2.63753,0.795119,1,16384
# Serial,13.0472,1.28588,1,65536
# SIMD,0.793108,21.1537,1,65536
# GPU,3.33076,5.03706,1,65536
# Serial,95.677,1.40282,1,262144
# SIMD,3.48052,38.5626,1,262144
# GPU,7.67591,17.4856,1,262144
# Serial,779.444,1.37757,1,1048576
# SIMD,16.9034,63.5222,1,1048576
# GPU,20.6245,52.0615,1,1048576

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("../gf2_test_results.csv")

# Calculate log2(sqrt(Matrix_Size))
df["Log2_Sqrt_Matrix_Size"] = np.log2(np.sqrt(df["Matrix_Size"]))

# Create three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

# Plot 1: Duration_ms vs log2(sqrt(Matrix_Size))
max_duration = df[df["Method"].isin(["SIMD", "GPU"])]["Duration_ms"].max()
y_limit = max_duration * 1.1  # 10% above the max value of SIMD and GPU

for method in df["Method"].unique():
    grouped = (
        df[df["Method"] == method]
        .groupby("Matrix_Size")["Duration_ms"]
        .agg(["mean", "std"])
    )
    ax1.errorbar(
        np.log2(np.sqrt(grouped.index)),
        grouped["mean"],
        yerr=grouped["std"],
        fmt="o-",
        label=method,
    )

ax1.set_xlabel("log2(sqrt(Matrix_Size))")
ax1.set_ylabel("Duration (ms)")
ax1.set_title("GF2 Multiplication Duration")
ax1.set_yscale("log")
ax1.legend()

# Plot 2: Throughput_GOPS vs log2(sqrt(Matrix_Size))
for method in df["Method"].unique():
    grouped = (
        df[df["Method"] == method]
        .groupby("Matrix_Size")["Throughput_GOPS"]
        .agg(["mean", "std"])
    )
    ax2.errorbar(
        np.log2(np.sqrt(grouped.index)),
        grouped["mean"],
        yerr=grouped["std"],
        fmt="o-",
        label=method,
    )

ax2.set_xlabel("log2(sqrt(Matrix_Size))")
ax2.set_ylabel("Throughput (GOPS)")
ax2.set_title("GF2 Multiplication Throughput")
ax2.legend()

# Plot 3: Throughput_GOPS vs log2(sqrt(Matrix_Size)) (Log Scale)
for method in df["Method"].unique():
    grouped = (
        df[df["Method"] == method]
        .groupby("Matrix_Size")["Throughput_GOPS"]
        .agg(["mean", "std"])
    )
    ax3.errorbar(
        np.log2(np.sqrt(grouped.index)),
        grouped["mean"],
        yerr=grouped["std"],
        fmt="o-",
        label=method,
    )

ax3.set_xlabel("log2(sqrt(Matrix_Size))")
ax3.set_ylabel("Throughput (GOPS)")
ax3.set_title("GF2 Multiplication Throughput (Log Scale)")
ax3.set_yscale("log")
ax3.legend()

plt.tight_layout()
plt.show()

print(df)

# Print grouped data for Duration_ms and Throughput_GOPS
print("\nGrouped Duration_ms (mean and std):")
for method in df["Method"].unique():
    grouped = (
        df[df["Method"] == method]
        .groupby("Matrix_Size")["Duration_ms"]
        .agg(["mean", "std"])
    )
    print(f"\nMethod: {method}")
    print(grouped)

print("\nGrouped Throughput_GOPS (mean and std):")
for method in df["Method"].unique():
    grouped = (
        df[df["Method"] == method]
        .groupby("Matrix_Size")["Throughput_GOPS"]
        .agg(["mean", "std"])
    )
    print(f"\nMethod: {method}")
    print(grouped)
