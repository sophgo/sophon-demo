#===----------------------------------------------------------------------===//
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===//
# WeChatQRCode QR detection + recognition (pure bmrt + bmcv, no sail).
#
# Build:
#   pcie: on an x86 host with libsophon + sophon-opencv
#     cmake -DTARGET_ARCH=pcie .. && make
#   soc:  cross-compile for aarch64 against a cross soc-sdk
#     cmake -DTARGET_ARCH=soc -DSDK=</path/to/soc-sdk> .. && make

cmake_minimum_required(VERSION 3.10)
project(wechat_qrcode_bmcv)

set(CMAKE_BUILD_TYPE Release)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

#Usage: check_and_replace_path(SDK)
function(check_and_replace_path ARG_NAME)
    if(IS_ABSOLUTE ${${ARG_NAME}})
        return()
    endif()
    set(PATH_TO_CHECK ${CMAKE_CURRENT_BINARY_DIR}/${${ARG_NAME}})
    if(EXISTS ${PATH_TO_CHECK})
        get_filename_component(ABSOLUTE_PATH ${PATH_TO_CHECK} ABSOLUTE)
        if(EXISTS ${ABSOLUTE_PATH})
            set(${ARG_NAME} ${ABSOLUTE_PATH} PARENT_SCOPE)
        else()
            message(FATAL_ERROR "Invalid path!")
        endif()
    else()
        message(FATAL_ERROR "Path ${PATH_TO_CHECK} does not exist")
    endif()
endfunction()

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

if (${TARGET_ARCH} STREQUAL "pcie")
    message("${TARGET_ARCH} mode, starting......")
    # use opencv (cv::imread / cvtColor / resize)
    set(OpenCV_DIR /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})
    # vendored zxing: add the subdirectory after the include dirs are registered
    # so its sources can find the OpenCV headers.
    add_subdirectory(zxing)
    # zxing includes opencv2/core/... headers, so pass the include dir explicitly.
    target_include_directories(zxing PUBLIC ${OpenCV_INCLUDE_DIRS})
    # use libbmlib libbmrt libbmcv
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/detector/*.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/scale/*.cpp)
    add_executable(wechat_qrcode_bmcv.pcie ${SRC_FILES})
    target_link_libraries(wechat_qrcode_bmcv.pcie zxing ${OpenCV_LIBS}
        ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} -lpthread)

elseif (${TARGET_ARCH} STREQUAL "soc")
    message("${TARGET_ARCH} mode, starting......")
    check_and_replace_path(SDK)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -std=c++14 -O2")
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs)
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    # The SDK's lib/ also ships a flat target-runtime dump that includes an
    # empty libstdc++.so placeholder. That placeholder shadows the cross
    # toolchain's C++ runtime for the driver's implicit -lstdc++, leaving
    # std::string symbols unresolved, so link the real libstdc++ by full path
    # (derived portably from the compiler instead of hardcoding a GCC dir).
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
                    OUTPUT_VARIABLE _libstdcxx OUTPUT_STRIP_TRAILING_WHITESPACE)
    link_directories("${SDK}/lib/")
    # vendored zxing: add the subdirectory here (after the cross compiler is
    # selected and the SDK include dirs registered) so its sources are compiled
    # with aarch64-linux-gnu-g++ and can find the SDK's OpenCV headers.
    add_subdirectory(zxing)
    # zxing includes opencv2/core/... headers, so pass the include dir explicitly.
    target_include_directories(zxing PUBLIC "${SDK}/include/opencv4" "${SDK}/include/")

    message("SDK: " ${SDK})

    file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/detector/*.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/scale/*.cpp)
    add_executable(wechat_qrcode_bmcv.soc ${SRC_FILES})
    target_link_libraries(wechat_qrcode_bmcv.soc zxing ${BM_LIBS} ${OPENCV_LIBS}
        "${_libstdcxx}" -lpthread)
else ()
    message(FATAL_ERROR "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()