#!/bin/bash
# build.sh — 编译 π0.5 C++ 例程（BM1684X SoC）
#
# 用法: ./build.sh [SOC_SDK 路径]
#   SOC_SDK 也可用环境变量指定；默认 /opt/sophon/sophon-sdk
#
# 产物: cpp/pi05_bmcv/pi05_bmcv.soc
# 交叉编译后 scp 到设备运行（见 cpp/pi05_bmcv/README.md）。

set -e

scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/..

SDK=${1:-${SOC_SDK:-/opt/sophon/sophon-sdk}}
if [ ! -d "$SDK" ]; then
    echo "SoC SDK 不存在: $SDK"
    echo "用法: $0 [SOC_SDK 路径]   或   export SOC_SDK=..."
    exit 1
fi

pushd $top_dir/cpp/pi05_bmcv
mkdir -p build && cd build
cmake .. -DTARGET_ARCH=soc -DSDK="$SDK"
make -j"$(nproc)"
popd

BIN=$top_dir/cpp/pi05_bmcv/pi05_bmcv.soc
echo "编译完成: $BIN"

# 交叉编译最容易踩的坑：编译过了，拿到设备上却起不来。宿主的 aarch64 交叉工具链往往
# 比设备新很多（例如 GCC 16 对 SE7 的 GCC 9 / glibc 2.31），链接出的二进制会要求设备上
# 不存在的 GLIBC/GLIBCXX 符号，运行时报 "version `GLIBCXX_3.4.32' not found"。
# 这里直接读二进制里记录的版本需求，与 SE7 的运行时对一下。
SE7_GLIBC=2.31          # Ubuntu 20.04
SE7_GLIBCXX=3.4.28
req_glibc=$(strings "$BIN" 2>/dev/null | grep -oE '^GLIBC_2\.[0-9]+$' | sort -V | tail -1)
req_glibcxx=$(strings "$BIN" 2>/dev/null | grep -oE '^GLIBCXX_3\.4\.[0-9]+$' | sort -V | tail -1)
echo "二进制要求: ${req_glibc:-?} / ${req_glibcxx:-?}   SE7 运行时提供: GLIBC_$SE7_GLIBC / GLIBCXX_$SE7_GLIBCXX"
if [ -n "$req_glibc" ] && [ "$(printf '%s\n%s\n' "$SE7_GLIBC" "${req_glibc#GLIBC_}" | sort -V | tail -1)" != "$SE7_GLIBC" ]; then
    worse=1
fi
if [ -n "$req_glibcxx" ] && [ "$(printf '%s\n%s\n' "$SE7_GLIBCXX" "${req_glibcxx#GLIBCXX_}" | sort -V | tail -1)" != "$SE7_GLIBCXX" ]; then
    worse=1
fi
if [ -n "${worse:-}" ]; then
    echo
    echo "WARNING: 这个二进制要求的 C/C++ 运行时比 SE7 上的新，拷过去会起不来"
    echo "         （报 GLIBC/GLIBCXX version not found）。"
    echo "         用与设备匹配的交叉工具链（SE7 是 GCC 9.x / glibc 2.31），"
    echo "         或者干脆在设备上原生编译："
    echo "             cd cpp/pi05_bmcv && mkdir -p build && cd build"
    echo "             cmake .. -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-current && make -j"
fi
