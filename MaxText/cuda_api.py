from ctypes import cdll, c_char_p, c_int, c_uint64, c_void_p
libcudart = cdll.LoadLibrary('libcudart.so')
def cudaProfilerStart():
    libcudart.cudaProfilerStart()
def cudaProfilerStop():
    libcudart.cudaProfilerStop()

libnvtx = cdll.LoadLibrary('libnvToolsExt.so')
libnvtx.nvtxRangePushA.argtypes = [c_char_p]
libnvtx.nvtxRangePushA.restype = c_int
libnvtx.nvtxRangeStartA.argtypes = [c_char_p]
libnvtx.nvtxRangeStartA.restype = c_uint64
libnvtx.nvtxRangeEnd.argtypes = [c_uint64]
libnvtx.nvtxDomainRangeStartEx.argtypes = [c_char_p, c_void_p]
def nvtxRangePush(name):
    return libnvtx.nvtxRangePushA(name.encode())
def nvtxRangePop():
    libnvtx.nvtxRangePop()
def nvtxRangeStart(name):
    return libnvtx.nvtxRangeStartA(name.encode())
def nvtxRangeEnd(range_id):
    libnvtx.nvtxRangeEnd(range_id)
def nvtxDomainRangeStart(domain, attrs):
    return libnvtx.nvtxRangeStartEx(domain, attrs)