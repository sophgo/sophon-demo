cmake_minimum_required(VERSION 3.10)
project(minicpm4)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/third_party/include)
include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    add_definitions(-DSOC_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/third_party/lib_soc)
    message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/third_party/lib_pcie)
    message("PCIE mode, starting......")
endif()


add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

aux_source_directory(. SRC_FILES)
add_executable(minicpm4 ${SRC_FILES})
target_link_libraries(minicpm4 PUBLIC bmrt bmlib tokenizers_cpp tokenizers_c sentencepiece pthread dl)
target_compile_options(minicpm4 PRIVATE -Wno-error=attributes)