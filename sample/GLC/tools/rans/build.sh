#!/bin/bash
# 编译 rANS 纯 C ABI 共享库，并跑冒烟测试。
#
# 源码只有 glc_rans.c + rans64.h，不依赖 torch / pybind11。
#
# 按当前机器架构自适应：
#   - 在 SE7 等 aarch64 设备上直跑：用本机 gcc 编出 aarch64 .so，无需交叉工具链。
#   - 在 x86_64 开发机上：本机 gcc 编 x86_64；若装了 aarch64-linux-gnu-gcc，
#     额外交叉编出 aarch64 版本，两架构码流比对一致。
#
# 两种场景都会跑同一份 rans_selftest.c 冒烟测试（能跑才算数，不只是能编译）。

set -euo pipefail

here=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
out_dir="${OUT_DIR:-$here}"
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

host_arch=$(uname -m)

build_lib() {
    local cc="$1" name="$2"
    echo "=== 编译 $name ==="
    "$cc" -O2 -fPIC -shared -I"$here" -o "$out_dir/$name" "$here/glc_rans.c"
}

build_selftest() {
    local cc="$1" out="$2"
    "$cc" -O2 -static -I"$here" -o "$out" "$here/glc_rans.c" "$here/rans_selftest.c"
}

# 取 selftest 输出的码流 hex，用于跨架构比对
run_selftest() {
    local runner="$1" bin="$2" label="$3"
    local output
    if [ -n "$runner" ]; then
        output=$("$runner" "$bin")
    else
        output=$("$bin")
    fi
    printf '%s\n' "$output" | grep '^encoded' | sed 's/^encoded [0-9]* bytes: //' > "$tmp_dir/$label.hex"
    printf '%s\n' "$output" | grep '^roundtrip:' | sed "s/^/$label /"
}

if [ "$host_arch" = "aarch64" ] || [ "$host_arch" = "arm64" ]; then
    # 设备上原生编译
    build_lib gcc libglc_rans_aarch64.so
    echo
    echo "=== 冒烟测试（本机 aarch64）==="
    build_selftest gcc "$tmp_dir/selftest_host"
    run_selftest "" "$tmp_dir/selftest_host" "aarch64"
    if ! grep -q . "$tmp_dir/aarch64.hex"; then
        echo "冒烟测试未产出码流" >&2
        exit 1
    fi
    echo "本机往返测试通过"
else
    # 开发机：本机 + 可选交叉
    build_lib gcc libglc_rans_x86_64.so
    echo
    echo "=== 冒烟测试（本机 x86_64）==="
    build_selftest gcc "$tmp_dir/selftest_host"
    run_selftest "" "$tmp_dir/selftest_host" "x86_64"

    if command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
        build_lib aarch64-linux-gnu-gcc libglc_rans_aarch64.so
        echo
        echo "=== 交叉编译 + 冒烟测试（aarch64）==="
        build_selftest aarch64-linux-gnu-gcc "$tmp_dir/selftest_arm"
        if command -v qemu-aarch64 >/dev/null 2>&1; then
            run_selftest qemu-aarch64 "$tmp_dir/selftest_arm" "aarch64"
        else
            run_selftest "" "$tmp_dir/selftest_arm" "aarch64"
        fi
        if ! cmp -s "$tmp_dir/x86_64.hex" "$tmp_dir/aarch64.hex"; then
            echo "两架构码流不一致" >&2
            diff "$tmp_dir/x86_64.hex" "$tmp_dir/aarch64.hex" >&2 || true
            exit 1
        fi
        echo "两架构码流逐字节一致"
    else
        echo
        echo "未装 aarch64-linux-gnu-gcc，跳过交叉编译。"
        echo "若要在 SE7 上使用，可直接在设备上跑本脚本（会原生编出 aarch64 版本）。"
    fi
fi

ls -la "$out_dir"/libglc_rans_*.so 2>/dev/null || true
