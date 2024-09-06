cmake_minimum_required(VERSION 3.10)
project(bytetrack_opencv C CXX)
SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")
if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

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

    #link our own include and srcs
    include_directories("../dependencies/include/")
    include_sub_directories_recursively(${CMAKE_SOURCE_DIR})
    file(GLOB_RECURSE SRC_FILES "./thirdparty/*.cpp") # dont involve build/
    aux_source_directory(. SRC_FILES)
    aux_source_directory(../dependencies/src/ SRC_FILES)
    add_executable(bytetrack_opencv.pcie ${SRC_FILES})
    target_link_libraries(bytetrack_opencv.pcie ${FFMPEG_LIBS} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} -lpthread)

elseif (${TARGET_ARCH} STREQUAL "soc")
    check_and_replace_path(SDK)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio opencv_video)
    include_directories("../dependencies/include/")
    include_sub_directories_recursively(${CMAKE_SOURCE_DIR})
    file(GLOB_RECURSE SRC_FILES "./thirdparty/*.cpp") # dont involve build/
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    link_directories("${SDK}/lib/")

    message(STATUS "SDK:" ${SDK})

    aux_source_directory(. SRC_FILES)
    aux_source_directory(../dependencies/src/ SRC_FILES)

    add_executable(bytetrack_opencv.soc ${SRC_FILES})
    target_link_libraries(bytetrack_opencv.soc ${BM_LIBS} ${OPENCV_LIBS} -lpthread -lavcodec -lavformat -lavutil)
else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()





