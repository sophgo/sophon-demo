cmake_minimum_required(VERSION 3.10)
project(yolov5)

set(CMAKE_CXX_FLAGS "-O3")
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

add_compile_definitions(CHIP_V5)
message( "${TARGET_ARCH} mode, starting......")
# set(lib_DIR /usr/lib/x84_64-linux-gnu)
# link_directories(${lib_DIR})
# use ffmpeg
set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
find_package(FFMPEG REQUIRED)
include_directories(${FFMPEG_INCLUDE_DIRS})
link_directories(${FFMPEG_LIB_DIRS})
# use opencv
set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIRS})
# use bmcv
include_directories(/opt/sophon/libsophav-latest/include)
link_directories(/opt/sophon/libsophav-latest/lib)

# use tpuv7-runtime
set(tpuv7_DIR /opt/tpuv7/tpuv7-current/data/)
find_package(tpuv7 REQUIRED)
include_directories(${TPUV7_INCLUDE_DIRS})
link_directories(${TPUV7_LIB_DIRS})

include_directories("../dependencies/include/")

aux_source_directory(. SRC_FILES)
aux_source_directory(../dependencies/src/ SRC_FILES)
add_executable(yolov5_bmcv ${SRC_FILES})
target_link_libraries(yolov5_bmcv ${FFMPEG_LIBS} ${OpenCV_LIBS} bmcv tpuv7_rt tpuv7_modelrt -lpthread)
