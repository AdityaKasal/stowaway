"""Free the disk space of byte ranges inside a file without changing its size or layout ("punching holes").

The ranges read back as zeros afterwards and take no space on disk. stowaway uses this to drop the expert data from
the original model file once it lives in the packed copy, so a model needs about its own size on disk, not twice it.

    punch(path, [(offset, length), ...])   -> bytes actually freed (ranges are shrunk to 4 KB boundaries)
    allocated(path)                        -> bytes the file really occupies on disk
"""

import ctypes
import os
import platform

BLOCK = 4096


def _aligned(ranges):
    for off, n in ranges:
        a = (off + BLOCK - 1) // BLOCK * BLOCK
        b = (off + n) // BLOCK * BLOCK
        if b > a:
            yield a, b - a


def allocated(path):
    if platform.system() == "Windows":
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCompressedFileSizeW.restype = ctypes.c_uint32
        high = ctypes.c_uint32(0)
        low = k32.GetCompressedFileSizeW(str(path), ctypes.byref(high))
        return (high.value << 32) + low
    return os.stat(path).st_blocks * 512


def punch(path, ranges):
    before = allocated(path)
    ranges = list(_aligned(ranges))
    system = platform.system()
    if system == "Linux":
        libc = ctypes.CDLL(None, use_errno=True)
        libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
        fd = os.open(path, os.O_RDWR)
        try:
            for off, n in ranges:
                if libc.fallocate(fd, 0x01 | 0x02, off, n) != 0:  # FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE
                    raise OSError(ctypes.get_errno(), f"fallocate punch-hole failed on {path}")
        finally:
            os.close(fd)
    elif system == "Darwin":
        import fcntl

        class FPunchhole(ctypes.Structure):
            _fields_ = [("fp_flags", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                        ("fp_offset", ctypes.c_int64), ("fp_length", ctypes.c_int64)]
        fd = os.open(path, os.O_RDWR)
        try:
            for off, n in ranges:
                fcntl.fcntl(fd, 99, bytes(FPunchhole(0, 0, off, n)))  # F_PUNCHHOLE
        finally:
            os.close(fd)
    elif system == "Windows":
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        h = k32.CreateFileW(str(path), 0x80000000 | 0x40000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)  # read|write, share all, OPEN_EXISTING
        if h in (None, wintypes.HANDLE(-1).value):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            got = wintypes.DWORD(0)
            if not k32.DeviceIoControl(h, 0x000900C4, None, 0, None, 0, ctypes.byref(got), None):  # FSCTL_SET_SPARSE
                raise ctypes.WinError(ctypes.get_last_error())
            for off, n in ranges:
                zero = (ctypes.c_int64 * 2)(off, off + n)  # FILE_ZERO_DATA_INFORMATION
                if not k32.DeviceIoControl(h, 0x000980C8, zero, 16, None, 0, ctypes.byref(got), None):  # FSCTL_SET_ZERO_DATA
                    raise ctypes.WinError(ctypes.get_last_error())
        finally:
            k32.CloseHandle(h)
    else:
        raise OSError(f"punching holes isn't supported on {system}")
    return max(0, before - allocated(path))


def make_sparse_file(path, size):
    """Create an empty file of `size` bytes that takes no disk space until written. On Windows the file is marked
    sparse first; otherwise NTFS would zero-fill up to wherever the first write lands (doubling the writes)."""
    path = str(path)
    with open(path, "wb"):
        pass
    if platform.system() == "Windows":
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        h = k32.CreateFileW(path, 0x80000000 | 0x40000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
        if h in (None, wintypes.HANDLE(-1).value):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            got = wintypes.DWORD(0)
            if not k32.DeviceIoControl(h, 0x000900C4, None, 0, None, 0, ctypes.byref(got), None):  # FSCTL_SET_SPARSE
                raise ctypes.WinError(ctypes.get_last_error())
            # set the size with the Windows API: Python's truncate() goes through the C runtime's _chsize, which
            # writes zeros into the extension and so allocates the whole file (measured: 1 GB for one 4 KB write)
            k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_int64, ctypes.c_void_p, wintypes.DWORD]
            if not k32.SetFilePointerEx(h, size, None, 0) or not k32.SetEndOfFile(h):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            k32.CloseHandle(h)
        return
    with open(path, "r+b") as f:
        f.truncate(size)
