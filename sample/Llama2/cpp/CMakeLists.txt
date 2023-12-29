cmake_minimum_required(VERSION 2.8)
project(llama2)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
if (${TARGET_ARCH} STREQUAL "pcie")
    message( "${TARGET_ARCH} mode, starting......")
    add_definitions(-DPCIE_TARGET)
    set(LIBSOPHON ${CMAKE_CURRENT_SOURCE_DIR}/../tools/libsophon-distributed/install/libsophon-0.4.9)
    include_directories(${LIBSOPHON}/include)
    link_directories(${LIBSOPHON}/lib) 

    include_directories(${PROJECT_SOURCE_DIR}/include)
    link_directories(${PROJECT_SOURCE_DIR}/lib_pcie)
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
    add_executable(llama2.pcie demo.cpp)
    target_link_libraries(llama2.pcie bmrt bmlib sentencepiece)

elseif (${TARGET_ARCH} STREQUAL "soc_base")
    message( "${TARGET_ARCH} mode, starting......")

    add_definitions(-DSOC_TARGET)
    set(LIBSOPHON /opt/sophon/libsophon-current)
    include_directories(${LIBSOPHON}/include)
    link_directories(${LIBSOPHON}/lib)

    include_directories(${PROJECT_SOURCE_DIR}/include)
    link_directories(${PROJECT_SOURCE_DIR}/lib_soc)
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
    add_executable(llama2.soc demo.cpp)
    target_link_libraries(llama2.soc pthread bmrt bmlib sentencepiece)

else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()
