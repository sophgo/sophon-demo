cmake_minimum_required(VERSION 3.10)
project(vlpr_bmcv C CXX)
include(${CMAKE_SOURCE_DIR}/../../../../cmake/common.cmake)
set(CMAKE_CXX_STANDARD 17)

# SET(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS "-O0")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

if (${TARGET_ARCH} STREQUAL "pcie")
    message( "${TARGET_ARCH} mode, starting......")
    # set(lib_DIR /usr/lib/x84_64-linux-gnu)
    # link_directories(${lib_DIR})
    # use ffmpeg
    set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})
    # use opencv
    set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4/)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})
    # use libbmrt libbmlib libbmcv
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    include_directories("./")
    include_directories("./yolov5_multi")
    include_directories("./lprnet_multi")

    aux_source_directory(. SRC_FILES)
    aux_source_directory(./lprnet_multi SRC_FILES)
    aux_source_directory(./yolov5_multi SRC_FILES)
    add_executable(vlpr_bmcv.pcie ${SRC_FILES})
    target_link_libraries(vlpr_bmcv.pcie ${FFMPEG_LIBS} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} -lpthread)

elseif (${TARGET_ARCH} STREQUAL "soc")
    message( "${TARGET_ARCH} mode, starting......")
    check_and_replace_path(SDK)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-8)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc-8)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-8)
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    link_directories("${SDK}/lib/")

    message(STATUS "SDK:" ${SDK})

    include_directories("./")
    include_directories("./yolov5_multi")
    include_directories("./lprnet_multi")

    aux_source_directory(. SRC_FILES)
    aux_source_directory(./lprnet_multi SRC_FILES)
    aux_source_directory(./yolov5_multi SRC_FILES)
    add_executable(vlpr_bmcv.soc ${SRC_FILES})
    target_link_libraries(vlpr_bmcv.soc ${BM_LIBS} ${OPENCV_LIBS}  ${JPU_LIBS} -lpthread -lavcodec -lavformat -lavutil -lstdc++fs)
else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()

# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


