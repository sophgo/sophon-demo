cmake_minimum_required(VERSION 3.10)
project(lprnet_bmcv C CXX)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH x86)
endif()

if (${TARGET_ARCH} STREQUAL "x86")
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
    # use libbmrt libbmlib libbmcv
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    include_directories("../../../../include/")

    aux_source_directory(. SRC_FILES)
    aux_source_directory(../../../../src/ SRC_FILES)

    add_executable(lprnet_bmcv.pcie ${SRC_FILES})
    target_link_libraries(lprnet_bmcv.pcie ${FFMPEG_LIBS} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} -lpthread)

elseif (${TARGET_ARCH} STREQUAL "soc")
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(BM_LIBS bmlib bmrt bmcv)
    set(JPU_LIBS bmjpuapi bmjpulite)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
    include_directories("../../../../include/")
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    link_directories("${SDK}/lib/")

    # message(STATUS "SDK:" ${SDK})

    aux_source_directory(. SRC_FILES)
    aux_source_directory(../../../../src/ SRC_FILES)
    add_executable(lprnet_bmcv.soc ${SRC_FILES})
    target_link_libraries(lprnet_bmcv.soc ${BM_LIBS} ${OPENCV_LIBS}  ${JPU_LIBS} -lpthread -lavcodec -lavformat -lavutil)

endif ()



