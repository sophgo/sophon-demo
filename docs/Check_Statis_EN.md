[简体中文](./Check_Statis.md) | [English](./Check_Statis_EN.md)

# Check Statistics

## 1. TPU Usage
You can use these two commands below:
```bash
bm-smi #Tpu-Util at top right corner indicates TPU's instant utilization, this command is available on both PCIe and SoC. On PCIe, SOPHON Boards, driver and libsophon is needed.
cat /sys/class/bm-tpu/bm-tpu0/device/npu_usage #Only SoC, it will print "usage:0 avusage:0", Usage indicates the NPU utilization within the past time window, and AvUsage indicates the NPU utilization since the driver was installed.
```
For more details, you can refer to 《LIBSOPHON Guide》。

## 2. Device Memory Usage
Check it by the field `Memory-Usage` at top right corner of command `bm-smi`, it represents the total number of GMEMs and the amount used; `bm-smi` also shows the amount of GMEM consumed by each process (or thread) on each device.

For more details, you can refer to 《LIBSOPHON Guide》。

## 3. CPU Usage
This refers to the utilization rate of a certain program, and the value of the corresponding program under the `CPU%` field can be viewed through `top` or `htop`, and users can search for their usage by themselves.

## 4. Host Mem Usage
This refers to the occupation of a certain program, and the value of the corresponding program under the 'RES' field can be viewed through 'top' or 'htop', and users can search for their usage by themselves.
