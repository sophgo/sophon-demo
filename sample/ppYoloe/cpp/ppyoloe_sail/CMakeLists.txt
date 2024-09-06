cmake_minimum_required(VERSION 3.12)
project(ppyoloe_sail)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "-O3")
add_definitions(-w)

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
    # set(lib_DIR /usr/lib/x84_64-linux-gnu)
    # link_directories(${lib_DIR})
    # use ffmpeg
    set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})
   # use opencv
    set(OpenCV_DIR /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4/)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    # use libbmrt libbmlib
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    include_directories("../dependencies/include/")

    # use sail
    set(SAIL_PATH /opt/sophon/sophon-sail)
    set(SAIL_LIBS  ${SAIL_PATH}/lib/libsail.so)
    message(STATUS ${SAIL_PATH})
    include_directories(${SAIL_PATH}/include/sail)

    add_compile_definitions(USE_BMCV=1)
    add_compile_definitions(USE_FFMPEG=1)
    add_compile_definitions(USE_OPENCV=1)

    # message(STATUS ${LIBSOPHON_LIB_DIRS})
    # message(STATUS ${LIBSOPHON_LIBS})
    # message(STATUS ${FFMPEG_LIB_DIRS})

    aux_source_directory(. SRC_FILES)
    add_executable(ppyoloe_sail.pcie ${SRC_FILES})
    target_link_libraries(ppyoloe_sail.pcie ${FFMPEG_LIBS} ${OpenCV_LIBS} ${SAIL_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} -lpthread)

elseif (${TARGET_ARCH} STREQUAL "soc")
    if (NOT DEFINED SDK)
        message(FATAL_ERROR "PLEASE SET SDK PATH")
    endif()
    check_and_replace_path(SDK)

    if (NOT DEFINED SAIL_PATH)
        message(FATAL_ERROR "PLEASE SET SAIL_PATH PATH")
    endif()
    check_and_replace_path(SAIL_PATH)

    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

    # set runtime sail
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH "/opt/sophon/sophon-sail/lib")

    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)

    include_directories("../dependencies/include/")

    message("SOC-SDK: ${SDK}")
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4/")
    link_directories("${SDK}/lib/")

    message("SAIL_PATH: ${SAIL_PATH}")
    include_directories("${SAIL_PATH}/include/sail")
    set(SAIL_LIBS "${SAIL_PATH}/lib/libsail.so")

    add_compile_definitions(USE_BMCV=1)
    add_compile_definitions(USE_FFMPEG=1)
    add_compile_definitions(USE_OPENCV=1)

    aux_source_directory(. SRC_FILES)
    add_executable(ppyoloe_sail.soc ${SRC_FILES})
    target_link_libraries(ppyoloe_sail.soc ${BM_LIBS} ${OPENCV_LIBS} ${SAIL_LIBS})
endif ()
