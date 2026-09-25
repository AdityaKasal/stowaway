"""Make this PC behave like one with less RAM: lock away memory so nothing else (including Windows' file cache) can
use it, leaving about LEAVE_GB available. Changes no settings; the memory comes back when this process exits.

usage: python ram_limit.py <leave_gb> <seconds>
"""

import ctypes
import ctypes.wintypes as wt
import sys
import time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
k32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.GetCurrentProcess.restype = wt.HANDLE
k32.SetProcessWorkingSetSizeEx.argtypes = [wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wt.DWORD]


class MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def avail():
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(MEMSTAT)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys


leave, seconds = float(sys.argv[1]) * 2**30, float(sys.argv[2])
take = int(avail() - leave) // (1 << 20) * (1 << 20)
print(f"available before: {avail() / 2**30:.1f} GB, locking {take / 2**30:.1f} GB", flush=True)
k32.SetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), take + (256 << 20), take + (512 << 20), 0)
chunk, locked = 1 << 30, 0
while locked < take:
    n = min(chunk, take - locked)
    p = k32.VirtualAlloc(None, n, 0x3000, 4)  # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
    if not p or not k32.VirtualLock(ctypes.c_void_p(p), n):
        print(f"stopped at {locked / 2**30:.1f} GB (error {ctypes.get_last_error()})", flush=True)
        break
    locked += n
print(f"locked {locked / 2**30:.1f} GB; available now: {avail() / 2**30:.1f} GB; holding for {seconds:.0f} s", flush=True)

# Windows slowly frees memory elsewhere (compression, trimming), which would loosen the limit; lock that too
end = time.time() + seconds
while time.time() < end:
    time.sleep(10)
    extra = int(avail() - leave - (256 << 20)) // (64 << 20) * (64 << 20)
    if extra > 0:
        k32.SetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), locked + extra + (256 << 20), locked + extra + (512 << 20), 0)
        p = k32.VirtualAlloc(None, extra, 0x3000, 4)
        if p and k32.VirtualLock(ctypes.c_void_p(p), extra):
            locked += extra
            print(f"locked {extra / 2**20:.0f} MB more (total {locked / 2**30:.1f} GB), available {avail() / 2**30:.1f} GB", flush=True)
