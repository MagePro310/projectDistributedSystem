import dask.array as da
from dask.distributed import Client

# Kết nối tới Scheduler
client = Client("tcp://192.168.1.105:8786")  # Thay bằng IP của Scheduler

# Hàm SUMMA với 2 Workers
def summa_two_workers(A, B):
    """
    SUMMA với 2 Workers
    :param A: Dask array, ma trận A
    :param B: Dask array, ma trận B
    :return: Ma trận kết quả C
    """
    # Kiểm tra kích thước
    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B, "Số cột của A phải bằng số hàng của B."

    # Chia khối: 2 hàng cho A, 2 cột cho B
    A = A.rechunk((M // 2, K))
    B = B.rechunk((K, N // 2))

    # Tạo ma trận kết quả
    C = da.zeros((M, N), chunks=(M // 2, N // 2))

    # SUMMA vòng lặp
    for k in range(2):  # Chỉ có 2 Workers
        # Chọn khối k từ A và B
        A_k = A[:, k * (K // 2):(k + 1) * (K // 2)]
        B_k = B[k * (K // 2):(k + 1) * (K // 2), :]

        # Tính toán và cộng dồn
        C += da.dot(A_k, B_k)

    return C

# Kích thước ma trận
M, K, N = 1000, 500, 800  # Giả sử chia đều được
chunk_size_M, chunk_size_K, chunk_size_N = M // 2, K // 2, N // 2

# Tạo ma trận Dask
A = da.random.random((M, K), chunks=(chunk_size_M, K))  # Chia A theo hàng
B = da.random.random((K, N), chunks=(K, chunk_size_N))  # Chia B theo cột

# Thực hiện SUMMA
C = summa_two_workers(A, B)

# Tính toán và thu kết quả
C_result = C.compute()

print("Phép nhân ma trận bằng SUMMA (2 Workers) hoàn tất!")
print(C_result)