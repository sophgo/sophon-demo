cmake_minimum_required(VERSION 3.10)
project(dinov2_bmcv)

# debug / release flags
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

SET(CMAKE_BUILD_TYPE "Release")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

#Usage: check_and_replace_path(SDK)
#Input src path, output absolute path.
function(check_and_replace_path ARG_NAME)
    if(IS_ABSOLUTE ${${ARG_NAME}})
        return()
    endif()
    set(PATH_TO_CHECK ${CMAKE_CURRENT_BINARY_DIR}/${${ARG_NAME}})
    if(EXISTS ${PATH_TO_CHECK})
        message("Path ${PATH_TO_CHECK} exists")
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
    message( "${TARGET_ARCH} mode, starting......")
    # use opencv（含 sophon-opencv 扩展：设备侧 imread / bmcv::toBMI）
    set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})
    # use libbmrt libbmlib
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    include_directories("../dependencies/include/")

    aux_source_directory(. SRC_FILES)
    add_executable(dinov2_bmcv.pcie ${SRC_FILES})
    target_link_libraries(dinov2_bmcv.pcie ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} -lpthread)


elseif (${TARGET_ARCH} STREQUAL "soc")
    message( "${TARGET_ARCH} mode, starting......")
    check_and_replace_path(SDK)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
    include_directories("../dependencies/include/")
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    link_directories("${SDK}/lib/")

    message("SDK: " ${SDK})

    aux_source_directory(. SRC_FILES)
    add_executable(dinov2_bmcv.soc ${SRC_FILES})
    if (EXISTS "${SDK}/lib/libstdc++.so.6")
        # 部分 soc-sdk（如 se9-soc-sdk）打包了 libstdc++ 运行库，但其中的 libstdc++.so
        # 是 0 字节占位文件，会顶掉工具链自动追加的 -lstdc++，导致 C++ 符号链接失败；
        # 此时按全路径直接链接 SDK 自带的运行时 libstdc++.so.6 以绕开空 stub。
        target_link_libraries(dinov2_bmcv.soc ${BM_LIBS} ${OPENCV_LIBS} ${JPU_LIBS} -lpthread "${SDK}/lib/libstdc++.so.6")
    else()
        # 未打包 libstdc++ 的 soc-sdk（如 SDK-26.03.01），交给交叉工具链自动链接。
        target_link_libraries(dinov2_bmcv.soc ${BM_LIBS} ${OPENCV_LIBS} ${JPU_LIBS} -lpthread)
    endif()
else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif()