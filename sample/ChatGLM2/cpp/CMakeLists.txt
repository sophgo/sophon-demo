
cmake_minimum_required(VERSION 2.8)
project(ChatGLM2)

# # set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_FLAGS "-O3")
# SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# message("cur dir: ${PROJECT_SOURCE_DIR}")
cmake_policy(SET CMP0015 NEW)
# cmake_policy(SET CMP0081 NEW)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})


if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()


if (${TARGET_ARCH} STREQUAL "pcie")
    message( "${TARGET_ARCH} mode, starting......")

    set(LIBSOPHON /opt/sophon/libsophon-current)
    include_directories(${LIBSOPHON}/include)
    link_directories(${LIBSOPHON}/lib)

    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/sentencepiece/lib/x86) 
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)

    add_executable(chatglm2.pcie main.cpp)
    target_link_libraries(chatglm2.pcie bmrt bmlib sentencepiece)


elseif (${TARGET_ARCH} STREQUAL "soc")
    message( "${TARGET_ARCH} mode, starting......")

    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    include_directories("${SDK}/include/")
    link_directories("${SDK}/lib/")

    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/sentencepiece/lib/arm64) 
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
    
    add_executable(chatglm2.soc main.cpp)
    target_link_libraries(chatglm2.soc bmrt bmlib sentencepiece)

elseif (${TARGET_ARCH} STREQUAL "soc_base")
    message( "${TARGET_ARCH} mode, starting......")

    set(LIBSOPHON /opt/sophon/libsophon-current)
    include_directories(${LIBSOPHON}/include)
    link_directories(${LIBSOPHON}/lib)

    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/sentencepiece/lib/arm64) 
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
    
    add_executable(chatglm2.soc main.cpp)
    target_link_libraries(chatglm2.soc bmrt bmlib sentencepiece dl pthread)

else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()