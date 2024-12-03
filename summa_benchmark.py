import dask.array as da
import time
import pandas as pd

# SUMMA implementation
def summa_two_workers(A, B):
    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B, "Number of columns of A must match rows of B."

    A = A.rechunk((M // 2, K))
    B = B.rechunk((K, N // 2))
    C = da.zeros((M, N), chunks=(M // 2, N // 2))

    for k in range(2):
        A_k = A[:, k * (K // 2):(k + 1) * (K // 2)]
        B_k = B[k * (K // 2):(k + 1) * (K // 2), :]
        C += da.dot(A_k, B_k)

    return C

# Benchmark function
def benchmark_summa(max_size, step):
    results = []

    for size in range(step, max_size + 1, step):
        M = K = N = size
        chunk_size = size // 2

        A = da.random.random((M, K), chunks=(chunk_size, K))
        B = da.random.random((K, N), chunks=(K, chunk_size))

        start_time = time.time()
        C = summa_two_workers(A, B)
        C.compute()
        end_time = time.time()

        computation_time = end_time - start_time
        results.append({"Matrix Size (n x n)": size, "Computation Time (s)": computation_time})
        print(f"Completed for size {size}x{size} in {computation_time:.4f} seconds")

    return pd.DataFrame(results)

# Run benchmark and save results
max_matrix_size = 10000
step_size = 500
benchmark_results = benchmark_summa(max_matrix_size, step_size)
benchmark_results.to_excel("summa_benchmark_results.xlsx", index=False)