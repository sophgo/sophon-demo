#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""libbmrt 极简 ctypes 绑定。

SE7 出厂镜像未预装 sophon-sail（SoC 版 wheel 不随 OTA 包分发），本文件用 ctypes
直接调用 libsophon 的 libbmrt / libbmlib，提供与 sail.Engine 等价的推理能力，
只依赖 numpy，无需编译。

用法:
    from bmrt import BmrtEngine
    eng = BmrtEngine("glc_encoder_fp32.bmodel")
    outs = eng.infer({"x": x, "q_enc": q})   # dict[str, np.ndarray]
"""
import ctypes
import os
import numpy as np

BM_MAX_DIMS_NUM = 8

BM_FLOAT32 = 0
BM_FLOAT16 = 1
BM_INT8 = 2
BM_UINT8 = 3
BM_INT16 = 4
BM_UINT16 = 5
BM_INT32 = 6
BM_UINT32 = 7
BM_BFLOAT16 = 8

_BM2NP = {
    BM_FLOAT32: np.float32,
    BM_FLOAT16: np.float16,
    BM_INT8: np.int8,
    BM_UINT8: np.uint8,
    BM_INT16: np.int16,
    BM_UINT16: np.uint16,
    BM_INT32: np.int32,
    BM_UINT32: np.uint32,
}
_NP2BM = {np.dtype(v): k for k, v in _BM2NP.items()}

LIB_DIRS = [
    "/opt/sophon/libsophon-current/lib",
    "/opt/sophon/libsophon-0.5.3/lib",
    "/usr/lib",
]


class BmrtError(RuntimeError):
    pass


class bm_shape_t(ctypes.Structure):
    _fields_ = [("num_dims", ctypes.c_int),
                ("dims", ctypes.c_int * BM_MAX_DIMS_NUM)]

    def to_tuple(self):
        return tuple(self.dims[: self.num_dims])

    @staticmethod
    def from_tuple(shape):
        s = bm_shape_t()
        s.num_dims = len(shape)
        for i, d in enumerate(shape):
            s.dims[i] = int(d)
        return s


class _bm_mem_device(ctypes.Structure):
    _fields_ = [("device_addr", ctypes.c_ulong),
                ("reserved", ctypes.c_uint),
                ("dmabuf_fd", ctypes.c_int)]


class _bm_mem_system(ctypes.Structure):
    _fields_ = [("system_addr", ctypes.c_void_p),
                ("reserved0", ctypes.c_uint),
                ("reserved1", ctypes.c_int)]


class _bm_mem_union(ctypes.Union):
    _fields_ = [("device", _bm_mem_device), ("system", _bm_mem_system)]


class bm_mem_flags_t(ctypes.Structure):
    # C 侧是位域 union {struct{...}; unsigned int rawflags;}，此处只需保证 4 字节
    _fields_ = [("rawflags", ctypes.c_uint)]


class bm_device_mem_t(ctypes.Structure):
    # struct bm_mem_desc { union {...} u; bm_mem_flags_t flags; unsigned int size; }
    # aarch64 下 sizeof = 24，务必与 C 侧一致，否则 bm_malloc_device_byte 会写越界
    _anonymous_ = ("u",)
    _fields_ = [("u", _bm_mem_union),
                ("flags", bm_mem_flags_t),
                ("size", ctypes.c_uint)]


class bm_tensor_t(ctypes.Structure):
    _fields_ = [("dtype", ctypes.c_int),
                ("shape", bm_shape_t),
                ("device_mem", bm_device_mem_t),
                ("st_mode", ctypes.c_int)]


class bm_stage_info_t(ctypes.Structure):
    _fields_ = [("input_shapes", ctypes.POINTER(bm_shape_t)),
                ("output_shapes", ctypes.POINTER(bm_shape_t)),
                ("input_mems", ctypes.POINTER(bm_device_mem_t)),
                ("output_mems", ctypes.POINTER(bm_device_mem_t))]


class bm_net_info_t(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("is_dynamic", ctypes.c_bool),
        ("input_num", ctypes.c_int),
        ("input_names", ctypes.POINTER(ctypes.c_char_p)),
        ("input_dtypes", ctypes.POINTER(ctypes.c_int)),
        ("input_scales", ctypes.POINTER(ctypes.c_float)),
        ("output_num", ctypes.c_int),
        ("output_names", ctypes.POINTER(ctypes.c_char_p)),
        ("output_dtypes", ctypes.POINTER(ctypes.c_int)),
        ("output_scales", ctypes.POINTER(ctypes.c_float)),
        ("stage_num", ctypes.c_int),
        ("stages", ctypes.POINTER(bm_stage_info_t)),
        ("max_input_bytes", ctypes.POINTER(ctypes.c_size_t)),
        ("max_output_bytes", ctypes.POINTER(ctypes.c_size_t)),
        ("input_zero_point", ctypes.POINTER(ctypes.c_int)),
        ("output_zero_point", ctypes.POINTER(ctypes.c_int)),
        ("input_loc_devices", ctypes.POINTER(ctypes.c_int)),
        ("output_loc_devices", ctypes.POINTER(ctypes.c_int)),
        ("core_num", ctypes.c_int),
        ("addr_mode", ctypes.c_int32),
    ]


_libs = None


# 与 libsophon 0.5.3 的 C 侧 sizeof 对齐（aarch64）。布局不符时 bmrt 调用会踩内存，
# 因此这里做一次硬校验，宁可提前报错也不要 segfault。
_EXPECTED_SIZES = {
    "bm_shape_t": 36,
    "bm_device_mem_t": 24,
    "bm_tensor_t": 72,
    "bm_net_info_t": 144,
}


def _check_layout():
    for name, want in _EXPECTED_SIZES.items():
        got = ctypes.sizeof(globals()[name])
        if got != want:
            raise BmrtError(
                "结构体 %s 尺寸 %d != 期望 %d，与当前 libsophon 头文件不匹配"
                % (name, got, want))
    if bm_tensor_t.device_mem.offset != 40:
        raise BmrtError("bm_tensor_t.device_mem 偏移 %d != 40"
                        % bm_tensor_t.device_mem.offset)


_check_layout()


def _load_libs():
    """加载 libbmlib / libbmrt。返回 (bmlib, bmrt) 两个 CDLL。"""
    global _libs
    if _libs is not None:
        return _libs

    found = {}
    for d in LIB_DIRS:
        for name in ("libbmlib.so", "libbmrt.so"):
            p = os.path.join(d, name)
            if name not in found and os.path.exists(p):
                found[name] = p
    missing = [n for n in ("libbmlib.so", "libbmrt.so") if n not in found]
    if missing:
        raise BmrtError(
            "找不到 %s，请确认已安装 libsophon 并设置 LD_LIBRARY_PATH" % ", ".join(missing))

    # libbmrt 依赖 libbmlib 的符号，先以 RTLD_GLOBAL 加载
    bmlib = ctypes.CDLL(found["libbmlib.so"], mode=ctypes.RTLD_GLOBAL)
    bmrt = ctypes.CDLL(found["libbmrt.so"], mode=ctypes.RTLD_GLOBAL)

    bmlib.bm_dev_request.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    bmlib.bm_dev_request.restype = ctypes.c_int
    bmlib.bm_dev_free.argtypes = [ctypes.c_void_p]
    bmlib.bm_dev_free.restype = None
    bmlib.bm_malloc_device_byte.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(bm_device_mem_t), ctypes.c_uint]
    bmlib.bm_malloc_device_byte.restype = ctypes.c_int
    bmlib.bm_free_device.argtypes = [ctypes.c_void_p, bm_device_mem_t]
    bmlib.bm_free_device.restype = None
    bmlib.bm_memcpy_s2d.argtypes = [ctypes.c_void_p, bm_device_mem_t, ctypes.c_void_p]
    bmlib.bm_memcpy_s2d.restype = ctypes.c_int
    bmlib.bm_memcpy_d2s.argtypes = [ctypes.c_void_p, ctypes.c_void_p, bm_device_mem_t]
    bmlib.bm_memcpy_d2s.restype = ctypes.c_int
    bmlib.bm_thread_sync.argtypes = [ctypes.c_void_p]
    bmlib.bm_thread_sync.restype = ctypes.c_int

    bmrt.bmrt_create.argtypes = [ctypes.c_void_p]
    bmrt.bmrt_create.restype = ctypes.c_void_p
    bmrt.bmrt_destroy.argtypes = [ctypes.c_void_p]
    bmrt.bmrt_destroy.restype = None
    bmrt.bmrt_load_bmodel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    bmrt.bmrt_load_bmodel.restype = ctypes.c_bool
    bmrt.bmrt_get_network_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
    bmrt.bmrt_get_network_name.restype = ctypes.c_char_p
    bmrt.bmrt_get_network_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    bmrt.bmrt_get_network_info.restype = ctypes.POINTER(bm_net_info_t)
    bmrt.bmrt_launch_tensor_ex.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(bm_tensor_t), ctypes.c_int,
        ctypes.POINTER(bm_tensor_t), ctypes.c_int,
        ctypes.c_bool, ctypes.c_bool]
    bmrt.bmrt_launch_tensor_ex.restype = ctypes.c_bool

    _libs = (bmlib, bmrt)
    return _libs


class _NetInfo(object):
    """从 bm_net_info_t 抽出 Python 侧描述。"""

    def __init__(self, info):
        self.name = info.contents.name.decode()
        self.input_num = info.contents.input_num
        self.output_num = info.contents.output_num
        self.stage_num = info.contents.stage_num
        self.core_num = info.contents.core_num
        stage = info.contents.stages[0]
        self.input_shapes = [stage.input_shapes[i].to_tuple()
                             for i in range(self.input_num)]
        self.output_shapes = [stage.output_shapes[i].to_tuple()
                              for i in range(self.output_num)]
        self.input_names = [info.contents.input_names[i].decode()
                            for i in range(self.input_num)]
        self.output_names = [info.contents.output_names[i].decode()
                             for i in range(self.output_num)]
        self.input_dtypes = [info.contents.input_dtypes[i]
                             for i in range(self.input_num)]
        self.output_dtypes = [info.contents.output_dtypes[i]
                              for i in range(self.output_num)]
        self.input_scales = [info.contents.input_scales[i]
                             for i in range(self.input_num)]
        self.output_scales = [info.contents.output_scales[i]
                              for i in range(self.output_num)]
        self.input_zero_point = [info.contents.input_zero_point[i]
                                 for i in range(self.input_num)]
        self.output_zero_point = [info.contents.output_zero_point[i]
                                  for i in range(self.output_num)]


def _to_device(bmlib, handle, arr):
    """把 host numpy 数组拷到设备内存，返回 bm_device_mem_t。"""
    mem = bm_device_mem_t()
    nbytes = arr.nbytes
    rc = bmlib.bm_malloc_device_byte(handle, ctypes.byref(mem), nbytes)
    if rc != 0:
        raise BmrtError("bm_malloc_device_byte 失败 (rc=%d, size=%d)" % (rc, nbytes))
    rc = bmlib.bm_memcpy_s2d(handle, mem, arr.ctypes.data_as(ctypes.c_void_p))
    if rc != 0:
        bmlib.bm_free_device(handle, mem)
        raise BmrtError("bm_memcpy_s2d 失败 (rc=%d)" % rc)
    return mem


class BmrtEngine(object):
    """单个 bmodel 的推理引擎（对应 bmodel 内的第 0 个 network）。"""

    def __init__(self, bmodel_path, dev_id=0):
        bmlib, bmrt = _load_libs()
        self._bmlib = bmlib
        self._bmrt = bmrt
        if not os.path.exists(bmodel_path):
            raise BmrtError("bmodel 不存在: %s" % bmodel_path)

        self.handle = ctypes.c_void_p()
        rc = bmlib.bm_dev_request(ctypes.byref(self.handle), dev_id)
        if rc != 0:
            raise BmrtError("bm_dev_request 失败 (rc=%d, dev_id=%d)" % (rc, dev_id))

        self.p_bmrt = bmrt.bmrt_create(self.handle)
        if not self.p_bmrt:
            bmlib.bm_dev_free(self.handle)
            raise BmrtError("bmrt_create 失败")

        if not bmrt.bmrt_load_bmodel(self.p_bmrt, bmodel_path.encode()):
            bmrt.bmrt_destroy(self.p_bmrt)
            bmlib.bm_dev_free(self.handle)
            raise BmrtError("bmrt_load_bmodel 失败: %s" % bmodel_path)

        self.bmodel_path = bmodel_path
        self.net_name = bmrt.bmrt_get_network_name(self.p_bmrt, 0).decode()
        info = bmrt.bmrt_get_network_info(self.p_bmrt, self.net_name.encode())
        if not info:
            raise BmrtError("bmrt_get_network_info 失败")
        self.info = _NetInfo(info)

    # ---- 量化辅助 ----
    @staticmethod
    def _quantize(arr, scale, zero_point, np_dtype):
        """float32 -> 定点。scale<=0 视为无需量化。"""
        if scale is None or scale <= 0:
            return arr.astype(np_dtype)
        q = np.round(arr / scale) + zero_point
        lo, hi = np.iinfo(np_dtype).min, np.iinfo(np_dtype).max
        return np.clip(q, lo, hi).astype(np_dtype)

    @staticmethod
    def _dequantize(arr, scale, zero_point):
        if scale is None or scale <= 0:
            return arr.astype(np.float32)
        return (arr.astype(np.float32) - zero_point) * scale

    def infer(self, feeds):
        """feeds: dict[str, np.ndarray]（按输入名给 float32 数组）。返回 dict[str, np.ndarray]。"""
        bmlib, bmrt = self._bmlib, self._bmrt
        info = self.info

        missing = [n for n in info.input_names if n not in feeds]
        if missing:
            raise BmrtError("缺少输入: %s（模型需要 %s）" % (missing, info.input_names))

        in_tensors = (bm_tensor_t * info.input_num)()
        out_tensors = (bm_tensor_t * info.output_num)()
        alloced = []

        try:
            for i, name in enumerate(info.input_names):
                arr = np.ascontiguousarray(feeds[name])
                want = tuple(info.input_shapes[i])
                if arr.shape != want:
                    raise BmrtError("输入 %s 形状 %s != 模型要求 %s"
                                    % (name, arr.shape, want))
                dt = info.input_dtypes[i]
                np_dt = _BM2NP[dt]
                if dt != BM_FLOAT32:
                    arr = self._quantize(arr.astype(np.float32),
                                         info.input_scales[i],
                                         info.input_zero_point[i], np_dt)
                else:
                    arr = arr.astype(np.float32)
                in_tensors[i].dtype = dt
                in_tensors[i].shape = bm_shape_t.from_tuple(arr.shape)
                in_tensors[i].st_mode = 0
                mem = _to_device(bmlib, self.handle, arr)
                alloced.append(mem)
                in_tensors[i].device_mem = mem

            ok = bmrt.bmrt_launch_tensor_ex(
                self.p_bmrt, self.net_name.encode(),
                in_tensors, info.input_num,
                out_tensors, info.output_num,
                False, False)
            if not ok:
                raise BmrtError("bmrt_launch_tensor_ex 失败 (net=%s)" % self.net_name)

            rc = bmlib.bm_thread_sync(self.handle)
            if rc != 0:
                raise BmrtError("bm_thread_sync 失败 (rc=%d)" % rc)

            outs = {}
            for i, name in enumerate(info.output_names):
                shape = out_tensors[i].shape.to_tuple()
                dt = out_tensors[i].dtype
                np_dt = _BM2NP[dt]
                host = np.empty(shape, dtype=np_dt)
                rc = bmlib.bm_memcpy_d2s(
                    self.handle, host.ctypes.data_as(ctypes.c_void_p),
                    out_tensors[i].device_mem)
                if rc != 0:
                    raise BmrtError("bm_memcpy_d2s 失败 (rc=%d, out=%s)" % (rc, name))
                if dt != BM_FLOAT32:
                    host = self._dequantize(host, info.output_scales[i],
                                            info.output_zero_point[i])
                outs[name] = host
            return outs
        finally:
            for mem in alloced:
                bmlib.bm_free_device(self.handle, mem)
            for i in range(info.output_num):
                # user_mem=False 时输出内存由 bmrt 分配，需自行释放
                if out_tensors[i].device_mem.u.device.device_addr:
                    bmlib.bm_free_device(self.handle, out_tensors[i].device_mem)

    def close(self):
        if getattr(self, "p_bmrt", None):
            self._bmrt.bmrt_destroy(self.p_bmrt)
            self.p_bmrt = None
        if getattr(self, "handle", None):
            self._bmlib.bm_dev_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
