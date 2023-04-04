cmake_minimum_required(VERSION 3.10)
project(bytetrack_bmcv C CXX)
SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")
if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

function(include_sub_directories_recursively root_dir)
    if (${root_dir} STREQUAL "${CMAKE_SOURCE_DIR}/build")
        message("skip include dir: " ${root_dir})
        return()
    endif()
    if (IS_DIRECTORY ${root_dir})               # 当前路径是一个目录吗，是的话就加入到包含目录
        message("include dir: " ${root_dir})
        include_directories(${root_dir})
    endif()

    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*) # 获得当前目录下的所有文件，让如ALL_SUB列表中
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})
            include_sub_directories_recursively(${root_dir}/${sub}) # 对子目录递归调用，包含
        endif()
    endforeach()
endfunction()

if (${TARGET_ARCH} STREQUAL "pcie")
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
    link_directories(${OpenCV_LIB_DIRS})
    include_directories(${OpenCV_INCLUDE_DIRS})
    # use libbmrt libbmlib
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    # use Eigen
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIRS})
    link_directories(${EIGEN3_LIB_DIRS})

    #link our own include and srcs
    include_directories("../../../../include/")
    include_sub_directories_recursively(${CMAKE_SOURCE_DIR})
    include_directories("../../../YOLOv5/cpp/yolov5_bmcv/")
    file(GLOB_RECURSE SRC_FILES "./thirdparty/*.cpp" "../../../YOLOv5/cpp/yolov5_bmcv/yolov5.cpp") # dont involve build/
    aux_source_directory(. SRC_FILES)
    aux_source_directory(../../../../src/ SRC_FILES)
    add_executable(bytetrack_bmcv.pcie ${SRC_FILES})
    target_link_libraries(bytetrack_bmcv.pcie ${FFMPEG_LIBS} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} -lpthread)

elseif (${TARGET_ARCH} STREQUAL "soc")
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(JPU_LIBS bmjpuapi bmjpulite)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
    include_directories("../../../../include/")
    include_directories(/usr/include/eigen3)
    include_sub_directories_recursively(${CMAKE_SOURCE_DIR})
    include_directories("../../../YOLOv5/cpp/yolov5_bmcv/")
    file(GLOB_RECURSE SRC_FILES "./thirdparty/*.cpp" "../../../YOLOv5/cpp/yolov5_bmcv/yolov5.cpp") # dont involve build/
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    link_directories("${SDK}/lib/")

    message(STATUS "SDK:" ${SDK})

    aux_source_directory(. SRC_FILES)
    aux_source_directory(../../../../src/ SRC_FILES)

    add_executable(bytetrack_bmcv.soc ${SRC_FILES})
    target_link_libraries(bytetrack_bmcv.soc ${BM_LIBS} ${OPENCV_LIBS} ${JPU_LIBS} -lpthread -lavcodec -lavformat -lavutil)
else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()





