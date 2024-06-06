
# 常见问题解答

## 目录

* [1 环境安装相关问题](#1-环境安装相关问题)


我们列出了一些用户和开发者在开发过程中会遇到的常见问题以及对应的解决方案，如果您发现了任何问题，请随时联系我们或创建相关issue，非常欢迎您提出的任何问题或解决方案。

## 1 环境安装相关问题
### 1.1 如果您遇到下面的这种问题，有可能是您的libsophon不匹配.
您可以拷贝本机的 `libsophon` 到您的 `lib_soc_bm1684x` 文件夹

```
[BMRT][load_bmodel:1704] INFO:Bmodel loaded, version 2.2+v1.7.beta.152-g10f3dc5e5-20240508
[BMRT][load_bmodel:1706] INFO:pre net num: 0, load net num: 83
[BMRT][load_tpu_module:1802] INFO:loading firmare in bmodel
[BMRT][preload_funcs:2121] INFO: core_id=0, multi_fullnet_func_id=30
[BMRT][preload_funcs:2124] INFO: core_id=0, dynamic_fullnet_func_id=31
[bmlib_memory][error] bm_alloc_gmem failed, dev_id = 0, size = 0x48a000
[BM_CHECK][error] BM_CHECK_RET fail /workspace/libsophon/bmlib/src/bmlib_memory.cpp: bm_malloc_device_byte_heap_mask: 1077
[BMRT][must_alloc_device_mem:2682] FATAL:device mem alloc failed: size=4759552[0x48a000] type_len=1 status=5 desc=io_mem
minicpm: /data/MiniCPM-2B-TPU/demo/demo.cpp:117: void MiniCPM::init(std::string, std::string, const std::vector<int>&): Assertion `true == ret' failed.
Aborted
```

![alt text](../pics/image.png)

```
cd /opt/sophon/libsophon-0.5.0/
cp libbmlib.so.0 ./MiniCPM-2B-TPU/support/lib_soc_bm1684x
cp libbmrt.so.1.0 ./MiniCPM-2B-TPU/support/lib_soc_bm1684x
```


