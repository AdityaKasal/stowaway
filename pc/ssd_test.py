"""Raw SSD read speed, reading the model file the way the expert cache does: unbuffered 4 MB reads, N threads."""
import ctypes, ctypes.wintypes as wt, random, sys, threading, time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wt.HANDLE
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]

class OVERLAPPED(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                ("Offset", wt.DWORD), ("OffsetHigh", wt.DWORD), ("hEvent", wt.HANDLE)]

path = sys.argv[1]
BLOCK = 4 << 20
size = __import__("os").path.getsize(path)

def run(n_threads, sequential, seconds=8):
    total = [0]
    lock = threading.Lock()
    stop = time.time() + seconds
    def worker(i):
        h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000 | 0x10000000, None)  # GENERIC_READ, NO_BUFFERING|RANDOM_ACCESS
        buf = k32.VirtualAlloc(None, BLOCK, 0x3000, 4)
        got = wt.DWORD()
        pos = (size // n_threads) * i // BLOCK * BLOCK
        n = 0
        while time.time() < stop:
            off = pos if sequential else random.randrange(0, size - BLOCK) // 4096 * 4096
            ov = OVERLAPPED(0, 0, off & 0xFFFFFFFF, off >> 32, None)
            k32.ReadFile(h, buf, BLOCK, ctypes.byref(got), ctypes.byref(ov))
            n += got.value
            pos += BLOCK
        with lock:
            total[0] += n
        k32.CloseHandle(h)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    return total[0] / (time.time() - t0) / 1e9

for seq in (True, False):
    for n in (1, 4, 16, 32):
        print(f"{'sequential' if seq else 'random    '} 4 MB reads, {n:2d} threads: {run(n, seq):5.2f} GB/s", flush=True)
