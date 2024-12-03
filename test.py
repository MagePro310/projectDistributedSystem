import numpy as np
from multiprocessing import Pool
import time

# Hàm nhân ma trận không dùng multiprocessing
def matrix_multiplication_normal(A, B):
    return np.dot(A, B)

# Hàm nhân ma trận với multiprocessing
def matrix_multiplication_worker(data):
    A, B, row = data
    return np.dot(A[row], B)

def parallel_matrix_multiplication(A, B):
    rows = A.shape[0]
    with Pool() as pool:
        results = pool.map(matrix_multiplication_worker, [(A, B, i) for i in range(rows)])
    return np.array(results)

# Hàm so sánh tốc độ
def compare_execution_time(matrix_size):
    A = np.random.randint(1, 10, size=(matrix_size, matrix_size))
    B = np.random.randint(1, 10, size=(matrix_size, matrix_size))

    # Đo thời gian nhân ma trận không dùng multiprocessing
    start_time_normal = time.time()
    result_normal = matrix_multiplication_normal(A, B)
    end_time_normal = time.time()
    time_normal = end_time_normal - start_time_normal

    # Đo thời gian nhân ma trận dùng multiprocessing
    start_time_parallel = time.time()
    result_parallel = parallel_matrix_multiplication(A, B)
    end_time_parallel = time.time()
    time_parallel = end_time_parallel - start_time_parallel

    # Kiểm tra tính chính xác
    is_correct = np.array_equal(result_normal, result_parallel)

    return time_normal, time_parallel, is_correct

# Thực hiện so sánh
if __name__ == "__main__":
    matrix_sizes = [100, 200, 500]  # Các kích thước ma trận để kiểm tra
    print(f"{'Matrix Size':<15}{'Time Normal (s)':<20}{'Time Parallel (s)':<20}{'Correct':<10}")
    for size in matrix_sizes:
        time_normal, time_parallel, is_correct = compare_execution_time(size)
        print(f"{size:<15}{time_normal:<20.5f}{time_parallel:<20.5f}{is_correct:<10}")
