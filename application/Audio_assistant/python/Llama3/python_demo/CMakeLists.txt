cmake_minimum_required(VERSION 3.10)
project(llama3)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/../support/include)

if (${TARGET_ARCH} STREQUAL "soc")
    add_definitions(-DSOC_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../support/lib_soc)
    message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../support/lib_pcie)
    message("PCIE mode, starting......")
endif()

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
# set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

foreach(CPP_FILE ${CPP_FILES})
    get_filename_component(MODULE_NAME ${CPP_FILE} NAME_WE)
    pybind11_add_module(${MODULE_NAME} ${CPP_FILE})
    target_link_libraries(${MODULE_NAME} PUBLIC bmrt bmlib)
    install(TARGETS ${MODULE_NAME} DESTINATION python3)
endforeach()

