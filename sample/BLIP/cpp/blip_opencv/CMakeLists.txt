cmake_minimum_required(VERSION 3.10)
project(blip_opencv LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

set(SRC_FILES main.cpp blip.cpp)

if (${TARGET_ARCH} STREQUAL "pcie")
    message("${TARGET_ARCH} mode, starting......")

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer
        ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/include
    )
    # use opencv
    set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})
    # use libbmrt libbmlib
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    add_executable(blip_opencv.pcie ${SRC_FILES})
    target_link_libraries(blip_opencv.pcie PUBLIC ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} utf8proc -lpthread)

    target_include_directories(blip_opencv.pcie PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc)

elseif (${TARGET_ARCH} STREQUAL "soc")
    message("${TARGET_ARCH} mode, starting......")
    check_and_replace_path(SDK)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -O2")

    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)

    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    link_directories("${SDK}/lib/")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/include)
    message("SDK: " ${SDK})

    add_executable(blip_opencv.soc ${SRC_FILES})
    target_link_libraries(blip_opencv.soc PUBLIC ${BM_LIBS} ${OPENCV_LIBS} utf8proc -lpthread)

    target_include_directories(blip_opencv.soc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc)

else ()
    message ( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()
