import math
import time
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster
import dask.array as da
from dask import delayed

# Start a local Dask cluster
cluster = LocalCluster()
client = Client(cluster)
print(client)

# Sequential matrix multiplication for blocks
@delayed
def seq_mult_block_delayed(A_block, B_block):
    """
    Delayed sequential matrix multiplication for blocks.
    :param A_block: NumPy array (block of matrix A).
    :param B_block: NumPy array (block of matrix B).
    :return: NumPy array (result of A_block * B_block).
    """
    M, K = A_block.shape
    K_B, N = B_block.shape
    assert K == K_B, "Inner dimensions must match for matrix multiplication."
    
    # Initialize result block
    C_block = np.zeros((M, N))
    
    # Perform basic ijk multiplication
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C_block[i, j] += A_block[i, k] * B_block[k, j]
    
    return C_block

# SUMMA with distributed computation
def summa_distributed(A, B, num_workers):
    """
    Distributed SUMMA algorithm using Dask.
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

    # Initialize list for delayed tasks
    delayed_results = []

    # Perform SUMMA iterations
    for k in range(num_workers):
        A_k = A[:, k * chunk_size_K:(k + 1) * chunk_size_K].compute()
        B_k = B[k * chunk_size_K:(k + 1) * chunk_size_K, :].compute()
        
        # Use delayed block multiplication
        delayed_results.append(seq_mult_block_delayed(A_k, B_k))

    # Combine results using sum
    C = da.from_delayed(delayed_results[0], shape=(M, N), dtype=np.float64)
    for delayed_result in delayed_results[1:]:
        C += da.from_delayed(delayed_result, shape=(M, N), dtype=np.float64)

    return C

# Benchmark function
def benchmark_summa_distributed(max_size, step):
    """
    Benchmark distributed SUMMA with fixed value matrices.
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
        C = summa_distributed(A, B, num_workers)
        C.compute()  # Trigger computation
        end_time = time.time()

        computation_time = end_time - start_time
        results.append({"Matrix Size (n x n)": size, "Computation Time (s)": computation_time})
        print(f"Completed for size {size}x{size} in {computation_time:.4f} seconds")

    return pd.DataFrame(results)

# Run benchmark
max_matrix_size = 2000  # Maximum size of the square matrix
step_size = 500         # Incremental step size
benchmark_results = benchmark_summa_distributed(max_matrix_size, step_size)

# Save results to an Excel file
output_file = "summa_distributed_benchmark_results.xlsx"
benchmark_results.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")