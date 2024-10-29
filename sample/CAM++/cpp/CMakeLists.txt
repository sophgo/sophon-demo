cmake_minimum_required(VERSION 3.10)
project(campplus)

set(CMAKE_CXX_FLAGS "-O3")
#set(CMAKE_BUILD_TYPE "Debug")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

# use libbmrt libbmlib
find_package(libsophon REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
link_directories(${LIBSOPHON_LIB_DIRS})

# use kaldi
include_directories("./dependencies/include/")
if (${TARGET_ARCH} STREQUAL "pcie")
    link_directories("./dependencies/lib_pcie_amd64")
elseif (${TARGET_ARCH} STREQUAL "soc")
    link_directories("./dependencies/lib_soc")
endif()

aux_source_directory(. SRC_FILES)
add_executable(campplus ${SRC_FILES})
target_link_libraries(campplus ${the_libbmlib.so} ${the_libbmrt.so} -lpthread -lstdc++ -lkaldi-feat -lkaldi-matrix -lkaldi-util -lkaldi-base -lm)
