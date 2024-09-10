cmake_minimum_required(VERSION 3.10)
project(wenet)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3") #increase performance significantly.
if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

if(TARGET_ARCH STREQUAL "pcie" )
    message( "${TARGET_ARCH} mode, starting......")
    SET(CMAKE_BUILD_TYPE "Debug") 
    # SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
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

    aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILES)
    add_executable(${PROJECT_NAME}.${TARGET_ARCH} main.cpp ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME}.${TARGET_ARCH} ${ARMADILLO_LIBRARIES} ${SNDFILE_LIBRARIES} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} ${CTCDECODE_CPP_DIR}/build/libctcdecode.so yaml-cpp)
    target_include_directories(${PROJECT_NAME}.${TARGET_ARCH} PRIVATE ${SNDFILE_INCLUDE_DIRS})
elseif (TARGET_ARCH STREQUAL "soc")
    message( "${TARGET_ARCH} mode, starting......")
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS sndfile opencv_calib3d opencv_core opencv_dnn opencv_features2d opencv_flann opencv_freetype opencv_highgui opencv_imgcodecs opencv_imgproc opencv_ml opencv_objdetect opencv_photo opencv_stitching opencv_video opencv_videoio)
    set(ARMADILLO_LIBRARIES libarmadillo.so)

    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
    message("cur dir: ${PROJECT_SOURCE_DIR}")
    include_directories(${PROJECT_SOURCE_DIR}/include)
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    link_directories("${SDK}/lib/")

    set(CTCDECODE_CPP_DIR ${PROJECT_SOURCE_DIR}/cross_compile_module/ctcdecode-cpp)
    include_directories(${CTCDECODE_CPP_DIR}/include)
    link_directories(${CTCDECODE_CPP_DIR}/build)

    set(3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/cross_compile_module/3rd_party)
    include_directories(${3RD_PARTY_DIR}/include)
    link_directories(${3RD_PARTY_DIR}/lib)

    aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILES)
    add_executable(${PROJECT_NAME}.${TARGET_ARCH} main.cpp ${SRC_FILES})
    message("ARMADILLO_LIBRARIES: ${ARMADILLO_LIBRARIES}")
    target_link_libraries(${PROJECT_NAME}.${TARGET_ARCH} ${ARMADILLO_LIBRARIES} ${BM_LIBS} 
    ${3RD_PARTY_DIR}/lib/blas/libblas.so.3 ${3RD_PARTY_DIR}/lib/lapack/liblapack.so.3 
    ${OPENCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so}  
    ${CTCDECODE_CPP_DIR}/build/libctcdecode.so ${CTCDECODE_CPP_DIR}/build/3rd_party/kenlm/lib/libkenlm.so ${CTCDECODE_CPP_DIR}/openfst-1.6.3/src/lib/.libs/libfst.so.8 yaml-cpp)
    target_include_directories(${PROJECT_NAME}.${TARGET_ARCH} PRIVATE ${SNDFILE_INCLUDE_DIRS})
else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()
