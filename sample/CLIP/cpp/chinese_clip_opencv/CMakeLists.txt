cmake_minimum_required(VERSION 3.10)
project(clip_opencv LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_BUILD_TYPE Debug)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

# Usage: check_and_replace_path(SDK)
# Input src path, output absolute path.
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
    set(SRC_FILES main.cpp clip/clip.cpp)

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer
        ${CMAKE_CURRENT_SOURCE_DIR}/clip
        ${CMAKE_CURRENT_SOURCE_DIR}/thirty_party
    )
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

    # use libbmrt libbmlib
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})
    
    add_executable(clip_opencv.pcie ${SRC_FILES})
    target_link_libraries(clip_opencv.pcie PUBLIC ${FFMPEG_LIBS} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} utf8proc)

    target_include_directories(clip_opencv.pcie PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc)

elseif (${TARGET_ARCH} STREQUAL "soc")
    message("${TARGET_ARCH} mode, starting......")
    check_and_replace_path(SDK)
    set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc-9)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++-9)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -std=c++11 -O2")

    set(CMAKE_CXX_STANDARD 17)  
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(JPU_LIBS bmjpuapi bmjpulite)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio opencv_dnn)

    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    link_directories("${SDK}/lib/")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/tokenizer")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/clip")
    message("SDK: " ${SDK})

    set(SRC_FILES main.cpp clip/clip.cpp)

    add_executable(clip_opencv.soc ${SRC_FILES})
    target_link_libraries(clip_opencv.soc PUBLIC ${BM_LIBS} ${OPENCV_LIBS} -lavcodec -lavformat -lavutil utf8proc)

    target_include_directories(clip_opencv.soc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc)

else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()