# dask distributed summa with fixed value matrices and dynamic chunk sizes
# terminal command: python summa_fixedvalue_dynamic.py
# scheduler run: dask scheduler
# worker run: dask worker tcp://<scheduler_ip>:8786
# another terminal: python summa_fixedvalue_dynamic.py
# watch the dask in: http://localhost:8787/status

import dask.array as da
from dask.distributed import Client
import time
import pandas as pd
import math
import numpy as np
from dask import delayed

# Connect to the Dask Scheduler
client = Client("tcp://192.168.250.213:8786")  # Replace with your scheduler address

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
def benchmark_summa_distributed(max_size, step_multiplier, num_repeats=3):
    """
    Benchmark distributed SUMMA with random value matrices, with size increasing by powers of step_multiplier.
    Runs each test multiple times and outputs results with separate columns for each run.
    
    :param max_size: Maximum matrix size (n x n).
    :param step_multiplier: Multiplier for each step (e.g., 2 for powers of 2).
    :param num_repeats: Number of times to repeat each test.
    :return: DataFrame containing size and computation times for each run.
    """
    results = []

    # Get the number of workers connected
    num_workers = len(client.scheduler_info()["workers"])
    print(f"Number of workers detected: {num_workers}")

    size = step_multiplier  # Start with the smallest size (2)
    while size <= max_size:
        M = K = N = size

        repeat_times = []  # Store times for each repeat
        for repeat in range(num_repeats):
            # Create matrices with random values between 1 and 10
            A = da.random.randint(1, 11, size=(M, K), chunks=(math.ceil(M / num_workers), K))
            B = da.random.randint(1, 11, size=(K, N), chunks=(K, math.ceil(N / num_workers)))

            # Measure time for computation
            start_time = time.time()
            C = summa_distributed(A, B, num_workers)
            C.compute()  # Trigger computation
            end_time = time.time()

            computation_time = end_time - start_time
            repeat_times.append(computation_time)
            print(f"Run {repeat + 1}/{num_repeats} for size {size}x{size} completed in {computation_time:.4f} seconds")

        # Add data to results
        result_entry = {
            "Matrix Size (n x n)": size,
        }
        for i, run_time in enumerate(repeat_times, start=1):
            result_entry[f"Run {i} Time (s)"] = run_time
        
        results.append(result_entry)

        size *= step_multiplier  # Multiply the size for the next step

    return pd.DataFrame(results)

# Run benchmark with step multiplier of 2 and 3 repeats for each size
max_matrix_size = 512  # Maximum size of the square matrix
step_multiplier = 2    # Increase size by powers of 2
num_repeats = 3        # Repeat each test 3 times

benchmark_results = benchmark_summa_distributed(max_matrix_size, step_multiplier, num_repeats)

# Save results to an Excel file
output_file = "summa_distributed_benchmark_results_repeated_separate_runs.xlsx"
benchmark_results.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")