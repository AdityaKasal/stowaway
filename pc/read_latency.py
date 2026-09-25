"""Per-read latency for expert-sized unbuffered reads (2.3 MB) with N threads, into many different buffers like the cache."""
import ctypes, ctypes.wintypes as wt, random, sys, threading, time
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wt.HANDLE
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
class OV(ctypes.Structure):
    _fields_ = [("a", ctypes.c_void_p), ("b", ctypes.c_void_p), ("Offset", wt.DWORD), ("OffsetHigh", wt.DWORD), ("h", wt.HANDLE)]
path, n_threads, seconds = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
LEN_ARG = int(sys.argv[4]) if len(sys.argv) > 4 else 2359296
import os
size = os.path.getsize(path)
LEN = LEN_ARG // 4096 * 4096  # read size, 4 KB multiple
lat = []
lock = threading.Lock()
def worker():
    h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000 | 0x10000000, None)
    bufs = [k32.VirtualAlloc(None, LEN, 0x3000, 4) for _ in range(64)]
    got = wt.DWORD(); mine = []
    stop = time.time() + seconds
    while time.time() < stop:
        off = random.randrange(0, size - LEN) // 4096 * 4096
        ov = OV(None, None, off & 0xFFFFFFFF, off >> 32, None)
        t0 = time.perf_counter()
        k32.ReadFile(h, random.choice(bufs), LEN, ctypes.byref(got), ctypes.byref(ov))
        mine.append(time.perf_counter() - t0)
    with lock: lat.extend(mine)
ts = [threading.Thread(target=worker) for _ in range(n_threads)]
t0 = time.time()
for t in ts: t.start()
for t in ts: t.join()
lat.sort()
print(f"{n_threads} threads: {len(lat)} reads, avg {1e3*sum(lat)/len(lat):.2f} ms, median {1e3*lat[len(lat)//2]:.2f} ms, "
      f"p90 {1e3*lat[int(len(lat)*.9)]:.2f} ms, total {len(lat)*LEN/1e9/(time.time()-t0):.2f} GB/s", flush=True)
