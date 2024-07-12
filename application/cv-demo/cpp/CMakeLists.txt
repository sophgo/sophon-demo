cmake_minimum_required(VERSION 3.10)
project(cvdemo)
include(${CMAKE_SOURCE_DIR}/../../../cmake/common.cmake)

# set(CMAKE_CXX_FLAGS "-O3")
# set(CMAKE_BUILD_TYPE "Debug")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH soc)
endif()

if (${TARGET_ARCH} STREQUAL "soc")
    message( "${TARGET_ARCH} mode, starting......")
    check_and_replace_path(SDK)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    link_directories("${SDK}/lib/")

    message("SDK: " ${SDK})

    aux_source_directory(. SRC_FILES)
    add_executable(cvdemo.soc ${SRC_FILES})

    target_link_libraries(cvdemo.soc ${BM_LIBS} ${OPENCV_LIBS} -lpthread -lavcodec -lavformat -lavutil -lavdevice -lavutil)
else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()
