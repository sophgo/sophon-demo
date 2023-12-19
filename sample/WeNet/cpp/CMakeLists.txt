cmake_minimum_required(VERSION 3.10)
project(wenet)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

if(TARGET_ARCH STREQUAL "pcie" OR TARGET_ARCH STREQUAL "soc")
    message( "${TARGET_ARCH} mode, starting......")
    SET(CMAKE_BUILD_TYPE "Debug") 
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    # SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

    # set(CMAKE_CXX_FLAGS "-O3")
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
    message("cur dir: ${PROJECT_SOURCE_DIR}")
    
    include_directories(${PROJECT_SOURCE_DIR}/include)

    # use libbmrt libbmlib
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    # use ffmpeg
    set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})

    # use opencv
    set(OpenCV_DIR /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    # use yaml-cpp
    find_package(yaml-cpp REQUIRED)
    include_directories(${yaml-cpp_INCLUDE_DIRS})

    find_package(Armadillo REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SNDFILE REQUIRED sndfile)

    set(CTCDECODE_CPP_DIR ${PROJECT_SOURCE_DIR}/ctcdecode-cpp)
    include_directories(${CTCDECODE_CPP_DIR}/include)
    include_directories("../../../include/")

    aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILES)
    add_executable(${PROJECT_NAME}.${TARGET_ARCH} main.cpp ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME}.${TARGET_ARCH} ${ARMADILLO_LIBRARIES} ${SNDFILE_LIBRARIES} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} ${CTCDECODE_CPP_DIR}/build/libctcdecode.so yaml-cpp)
    target_include_directories(${PROJECT_NAME}.${TARGET_ARCH} PRIVATE ${SNDFILE_INCLUDE_DIRS})
else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()
