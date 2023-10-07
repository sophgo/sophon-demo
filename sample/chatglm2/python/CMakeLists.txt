cmake_minimum_required(VERSION 2.8)
project(ChatGLM2)

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_FLAGS "-O3")
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0081 NEW)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)


if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()


if (${TARGET_ARCH} STREQUAL "pcie")
    message( "${TARGET_ARCH} mode, starting......")

    set(LIBSOPHON /opt/sophon/libsophon-current)
    include_directories(${LIBSOPHON}/include)
    link_directories(${LIBSOPHON}/lib)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/lib_pcie) 

    set(PYBIND11_PYTHON_VERSION 3.7)
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
    set(pybind11_DIR "/usr/local/lib/python3.8/dist-packages/pybind11/share/cmake/pybind11")
    find_package(pybind11 REQUIRED CONFIG)
    pybind11_add_module(ChatGLM2 pybind.cpp)
    
    target_link_libraries(ChatGLM2 bmrt bmlib sentencepiece)
    install(TARGETS ChatGLM2 DESTINATION python)


elseif (${TARGET_ARCH} STREQUAL "soc")
    message( "${TARGET_ARCH} mode, starting......")

    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    
    include_directories("${SDK}/include/")
    link_directories("${SDK}/lib/")

    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/lib_soc) 

   
    set(PYBIND11_PYTHON_VERSION 3.7)
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
   
     # 设置源文件和目标路径
     set(PYBIND11_PYTHON_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/lib_soc/aarch64-linux-gnu)

     set(source_file ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/lib_soc/aarch64-linux-gnu)
     set(target_link /usr/include/aarch64-linux-gnu)
 
     # 软链
     execute_process(
         COMMAND sudo ln -s ${source_file} ${target_link}
     )
 
    set(pybind11_DIR "/usr/local/lib/python3.8/dist-packages/pybind11/share/cmake/pybind11")
    find_package(pybind11 REQUIRED CONFIG)
    pybind11_add_module(ChatGLM2 pybind.cpp)

    target_link_libraries(ChatGLM2 bmrt bmlib sentencepiece)
    install(TARGETS ChatGLM2 DESTINATION python)


elseif (${TARGET_ARCH} STREQUAL "soc_base")
   message( "${TARGET_ARCH} mode, starting......")


    set(LIBSOPHON /opt/sophon/libsophon-current)
    include_directories(${LIBSOPHON}/include)
    link_directories(${LIBSOPHON}/lib)

    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/lib_soc) 

    set(PYBIND11_PYTHON_VERSION 3.7)
    
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)

    set(pybind11_DIR "/home/linaro/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")
    
    find_package(pybind11 REQUIRED CONFIG)
    pybind11_add_module(ChatGLM2 pybind.cpp)
    
    target_link_libraries(ChatGLM2 bmrt bmlib sentencepiece)
    install(TARGETS ChatGLM2 DESTINATION python)

else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()