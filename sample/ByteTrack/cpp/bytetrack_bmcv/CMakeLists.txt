cmake_minimum_required(VERSION 3.5)
project(bytetrack-bmcv-cpp)
set(CMAKE_CXX_STANDARD 11)

set(lib_DIR /usr/lib/x84_64-linux-gnu)
link_directories(${lib_DIR})
include_directories(${FFMPEG_INCLUDE_DIRS})
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavdevice
    libavfilter
    libavformat
    libavcodec
    libswresample
    libswscale
    libavutil
)
include_directories(include)
# use opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# use eigen3
include_directories(/usr/local/include/eigen3)
# use libbmrt libbmlib
find_package(libsophon REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
link_directories(${LIBSOPHON_LIB_DIRS})
# use sail
set(SAIL_LIBS  ${SAIL_DIR}/lib/libsail.so)
message(STATUS ${SAIL_DIR})
include_directories(${SAIL_DIR}/include)
include_directories(${SAIL_DIR}/include/sail)

message(STATUS ${LIBSOPHON_LIB_DIRS})
message(STATUS ${LIBSOPHON_LIBS})
message(STATUS ${FFMPEG_LIB_DIRS})

file(GLOB My_Source_Files src/*.cpp)
add_executable(bytetrack-bmcv-cpp ${My_Source_Files})
target_link_libraries(bytetrack-bmcv-cpp ${OpenCV_LIBS} ${FFMPEG_LIBS} ${SAIL_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so})