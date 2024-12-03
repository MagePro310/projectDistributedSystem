# File: scheduler.py
from dask.distributed import Scheduler

def main():
    # Scheduler lắng nghe trên địa chỉ IP và cổng
    scheduler = Scheduler(host="192.168.1.105", port=8786)  # Thay IP bằng IP máy gốc
    print("Scheduler is running on:", scheduler.address)
    scheduler.start()

if __name__ == "__main__":
    main()
