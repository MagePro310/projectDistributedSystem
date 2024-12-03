import dask.array as da
from dask.distributed import Client
import time
import pandas as pd
import math

# Connect to the Dask Scheduler
client = Client("tcp://192.168.1.105:8786")  # Replace with your scheduler address

# SUMMA implementation with uneven chunks
def summa_dynamic_uneven(A, B, num_workers):
    """
    SUMMA algorithm dynamically scaled for any number of workers, handling uneven chunks.
    :param A: Dask array, matrix A.
    :param B: Dask array, matrix B.
    :param num_workers: Number of workers connected to the Dask cluster.
    :return: Matrix C (result of A * B).
    """
    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B, "Number of columns in A must match rows in B."

    # Calculate chunk sizes
    chunk_size_M = math.ceil(M / num_workers)
    chunk_size_K = math.ceil(K / num_workers)
    chunk_size_N = math.ceil(N / num_workers)

    # Rechunk matrices
    A = A.rechunk((chunk_size_M, K))
    B = B.rechunk((K, chunk_size_N))
    C = da.zeros((M, N), chunks=(chunk_size_M, chunk_size_N))

    # Perform SUMMA iterations
    for k in range(num_workers):
        A_k = A[:, k * chunk_size_K:(k + 1) * chunk_size_K]
        B_k = B[k * chunk_size_K:(k + 1) * chunk_size_K, :]
        C += da.dot(A_k, B_k)

    return C

# Benchmark function
def benchmark_summa_fixed_value(max_size, step):
    """
    Benchmark SUMMA with fixed value matrices and dynamic scaling.
    :param max_size: Maximum matrix size (n x n).
    :param step: Step size for increasing matrix dimensions.
    :return: DataFrame containing size and computation times.
    """
    results = []

    # Get the number of workers connected
    num_workers = len(client.scheduler_info()["workers"])
    print(f"Number of workers detected: {num_workers}")

    for size in range(step, max_size + 1, step):
        M = K = N = size
        value = 3  # Fixed value for all matrix elements

        # Create matrices with fixed values
        A = da.full((M, K), value, chunks=(math.ceil(M / num_workers), K))
        B = da.full((K, N), value, chunks=(K, math.ceil(N / num_workers)))

        # Measure time for computation
        start_time = time.time()
        C = summa_dynamic_uneven(A, B, num_workers)
        C.compute()  # Trigger computation
        end_time = time.time()

        computation_time = end_time - start_time
        results.append({"Matrix Size (n x n)": size, "Computation Time (s)": computation_time})
        print(f"Completed for size {size}x{size} in {computation_time:.4f} seconds")

    return pd.DataFrame(results)

# Run benchmark
max_matrix_size = 2000  # Maximum size of the square matrix
step_size = 500         # Incremental step size
benchmark_results = benchmark_summa_fixed_value(max_matrix_size, step_size)

# Save results to an Excel file
output_file = "summa_fixed_value_benchmark_results.xlsx"
benchmark_results.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")
