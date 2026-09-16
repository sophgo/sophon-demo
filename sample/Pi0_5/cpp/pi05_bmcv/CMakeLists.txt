cmake_minimum_required(VERSION 3.10)
project(pi05)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# π0.5 例程只支持 BM1684X 的 SoC 模式（SE7 系列）。
# 需要先交叉编译，再 scp 到设备运行；编译产物后缀 .soc。
#
#   cmake .. -DTARGET_ARCH=soc -DSDK=<soc-sdk 绝对路径>
#   make -j
#
# SoC SDK 路径也可用环境变量 SOC_SDK 指定。
#
# 也可以直接在设备上原生编译（SE7 自带 g++ 9.x 与 libsophon，可跳过交叉工具链）：
#   cmake .. -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-current
#   make -j

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH soc)
endif()

if (NOT ${TARGET_ARCH} STREQUAL "soc")
    message(FATAL_ERROR "only TARGET_ARCH=soc is supported (BM1684X / SE7)")
endif()

if (NOT DEFINED SDK)
    if (DEFINED ENV{SOC_SDK})
        set(SDK $ENV{SOC_SDK})
    else()
        message(FATAL_ERROR "please pass -DSDK=<soc-sdk path> or set SOC_SDK")
    endif()
endif()
if (NOT EXISTS "${SDK}")
    message(FATAL_ERROR "SDK path does not exist: ${SDK}")
endif()

# BM1684X 用 g++ 9.x（BM1688 / CV186X 才是 11.x）
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)

# libsophon is the only dependency: the example reads its observations from .npy arrays,
# so it needs no image decoding library (bmlib / bmrt cover device access and inference).
set(BM_LIBS bmlib bmrt)

include_directories("${SDK}/include/")
link_directories("${SDK}/lib/")
include_directories(".")

aux_source_directory(. SRC_FILES)
add_executable(pi05_bmcv.soc ${SRC_FILES})
target_link_libraries(pi05_bmcv.soc ${BM_LIBS} -lpthread)
