#===----------------------------------------------------------------------===//
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===//
# WeChatQRCode QR detection + recognition via the sail delivery API.
#
# Only the thin demo is compiled here; the whole pipeline lives in libsail.so
# (sail::wechat_qrcode). SAIL_PATH is a sail source/build tree: the public
# headers live directly under include/ (e.g. wechat_qrcode.hpp) and their
# third-party dependency headers (spdlog, pulled in by tensor.h) under
# 3rdparty/.
#
# Build:
#   pcie: on an x86 host with libsophon + sophon-opencv + sophon-sail
#     cmake -DTARGET_ARCH=pcie .. && make
#   soc:  cross-compile for aarch64 against a cross soc-sdk + aarch64 libsail.so
#     cmake -DTARGET_ARCH=soc -DSDK=</path/to/soc-sdk> \
#           -DSAIL_PATH=</path/to/sail> .. && make

cmake_minimum_required(VERSION 3.10)
project(wechat_qrcode_sail)

set(CMAKE_BUILD_TYPE Release)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# sail's public headers gate BMImage/Bmcv/Decoder behind USE_BMCV/USE_OPENCV/
# USE_FFMPEG. The default decode path is sail::Decoder (USE_OPENCV_DECODE=0 in
# main.cpp), which needs USE_FFMPEG (cvwrapper.h pulls in the ffmpeg headers);
# USE_OPENCV is kept for the fallback cv::imread + Bmcv::mat_to_bm_image branch.
add_compile_definitions(USE_BMCV=1 USE_OPENCV=1 USE_FFMPEG=1)

#Usage: check_and_replace_path(SDK)
#Input src path, output absolute path.
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
    # use ffmpeg (sail::Decoder hardware decode path)
    set(FFMPEG_DIR /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})
    # use opencv (cv::imread in the demo; sail uses it too)
    set(OpenCV_DIR /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})
    # use libbmlib libbmrt libbmcv
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    # use sail (full libsail.so carries sail::wechat_qrcode symbols)
    set(SAIL_PATH /opt/sophon/sophon-sail)
    set(SAIL_LIBS ${SAIL_PATH}/lib/libsail.so)
    message(STATUS "SAIL_PATH: ${SAIL_PATH}")
    include_directories(${SAIL_PATH}/include)
    include_directories(${SAIL_PATH}/3rdparty)

    aux_source_directory(. SRC_FILES)
    add_executable(wechat_qrcode_sail.pcie ${SRC_FILES})
    target_link_libraries(wechat_qrcode_sail.pcie ${FFMPEG_LIBS} ${OpenCV_LIBS}
        ${SAIL_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} -lpthread)

elseif (${TARGET_ARCH} STREQUAL "soc")
    message("${TARGET_ARCH} mode, starting......")
    check_and_replace_path(SDK)
    check_and_replace_path(SAIL_PATH)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -std=c++14 -O2")
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs)

    message("SOC-SDK: ${SDK}")
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4/")
    # The SDK's lib/ also ships a flat target-runtime dump that includes an
    # empty libstdc++.so placeholder. That placeholder shadows the cross
    # toolchain's C++ runtime for the driver's implicit -lstdc++, leaving
    # std::string symbols unresolved, so link the real libstdc++ by full path
    # (derived portably from the compiler instead of hardcoding a GCC dir).
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
                    OUTPUT_VARIABLE _libstdcxx OUTPUT_STRIP_TRAILING_WHITESPACE)
    link_directories("${SDK}/lib/")

    message("SAIL_PATH: ${SAIL_PATH}")
    include_directories("${SAIL_PATH}/include")
    include_directories("${SAIL_PATH}/3rdparty")
    set(SAIL_LIBS "${SAIL_PATH}/lib/libsail.so")

    aux_source_directory(. SRC_FILES)
    add_executable(wechat_qrcode_sail.soc ${SRC_FILES})
    # --allow-shlib-undefined: undefined refs inside libsail.so (bmrt_*/bm_image_*)
    # are provided by the target machine at runtime.
    target_link_libraries(wechat_qrcode_sail.soc ${BM_LIBS} ${OPENCV_LIBS}
        ${SAIL_LIBS} -lpthread -lavcodec -lavformat -lavutil
        "${_libstdcxx}" -Wl,--allow-shlib-undefined)
else ()
    message(FATAL_ERROR "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()