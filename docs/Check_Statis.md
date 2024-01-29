[简体中文](./Check_Statis.md) | [English](./Check_Statis_EN.md)

# 常用指标查看方法

## 1. TPU利用率
可以通过以下两种命令查看：
```bash
bm-smi #右上角的Tpu-Util表示TPU瞬时利用率，PCIe和SoC均可使用，PCIe下需要搭载SOPHON板卡并安装驱动和LIBSOPHON。
cat /sys/class/bm-tpu/bm-tpu0/device/npu_usage #SoC下可以使用，打印usage:0 avusage:0，Usage表示过去一个时间窗口内的npu利用率，AvUsage表示自安装驱动以来npu的利用率。
```
更多详细信息，可以参考《LIBSOPHON 使用手册》。

## 2. 设备内存占用
可以通过`bm-smi`右上角的`Memory-Usage`查看，它表示gmem总数和已使用数量；`bm-smi`还会显示每个设备上每个进程（或者线程）占用的gmem的数量。

更多详细信息，可以参考《LIBSOPHON 使用手册》。

## 3. CPU利用率
这里指的是某个程序的利用率，可以通过`top`或者`htop`查看`CPU%`字段下对应程序的值，用户可以自行搜索它们的用法。

## 4. 系统内存占用
这里指的是某个程序的占用，可以通过`top`或者`htop`查看`RES`字段下对应程序的值，用户可以自行搜索它们的用法。