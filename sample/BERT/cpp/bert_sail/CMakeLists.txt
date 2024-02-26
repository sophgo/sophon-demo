
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(bert_sail)
include(${CMAKE_SOURCE_DIR}/../../../../cmake/common.cmake)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3")

add_definitions(-w)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()
if (${TARGET_ARCH} STREQUAL "pcie")
    message( "${TARGET_ARCH} mode, starting......")
    set(lib_DIR /usr/lib/x84_64-linux-gnu)
    link_directories(${lib_DIR})
   
    include_directories(libtorch_tokenizer/src/unilib)
    include_directories(libtorch_tokenizer/src)
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})
    # 添加libsophon的头文件目录
    include_directories("../../../../include/")

    set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})
    # 添加sophon-ffmpeg的头文件及动态库目录
    set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    # 添加sophon-opencv的头文件目录
    set(SAIL_DIR  /opt/sophon/sophon-sail/lib/cmake)
    find_package(SAIL REQUIRED)
    include_directories(${SAIL_INCLUDE_DIRS})
    link_directories(${SAIL_LIB_DIRS})
    # 添加SAIL的头文件及动态库目录
    
    add_executable(bert_sail.pcie libtorch_tokenizer/src/tokenizer.cpp libtorch_tokenizer/src/unilib/unicode.cpp libtorch_tokenizer/src/unilib/uninorms.cpp bert_sail.cpp bert_sail.hpp main.cpp)
    target_link_libraries(bert_sail.pcie sail ${FFMPEG_LIBS} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} )

    target_link_libraries(bert_sail.pcie ${TORCH_LIBRARIES})
    
elseif (${TARGET_ARCH} STREQUAL "soc")
    message( "${TARGET_ARCH} mode, starting......")
    check_and_replace_path(SDK)
    check_and_replace_path(SAIL_PATH)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    
    include_directories(libtorch_tokenizer/src/unilib)
    include_directories(libtorch_tokenizer/src)
    set(BM_LIBS bmlib bmrt bmcv)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    

    link_directories("${SDK}/lib/")
    include_directories("../../../../include/")
    # use sail headers
    include_directories(${SAIL_PATH}/include)
    include_directories(${SAIL_PATH}/include/sail)
    # use sail
    set(SAIL_LIBS  ${SAIL_PATH}/lib/libsail.so)
    # sail为需要链接的库
    message("SDK: " ${SDK})

    add_executable(bert_sail.soc libtorch_tokenizer/src/tokenizer.cpp libtorch_tokenizer/src/unilib/unicode.cpp libtorch_tokenizer/src/unilib/uninorms.cpp bert_sail.cpp bert_sail.hpp main.cpp)
    target_link_libraries(bert_sail.soc ${BM_LIBS} ${OPENCV_LIBS}  ${JPU_LIBS} ${SAIL_LIBS})
  
    target_link_libraries(bert_sail.soc ${TORCH_LIBRARIES})
else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()