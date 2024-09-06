
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(bert_sail)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3")

add_definitions(-w)

#Usage: check_and_replace_path(SDK)
#Input src path, output absolute path.
function(check_and_replace_path ARG_NAME) 
    if(IS_ABSOLUTE ${${ARG_NAME}})
        return()
    endif()
    set(PATH_TO_CHECK ${CMAKE_CURRENT_BINARY_DIR}/${${ARG_NAME}})
    if(EXISTS ${PATH_TO_CHECK})
        message("Path ${PATH_TO_CHECK} exists")
        get_filename_component(ABSOLUTE_PATH ${PATH_TO_CHECK} ABSOLUTE)
        if(EXISTS ${ABSOLUTE_PATH})
            set(${ARG_NAME} ${ABSOLUTE_PATH} PARENT_SCOPE)
        else()
            message(FATAL_ERROR "Invalid path!")
        endif()
    else()
        message(FATAL_ERROR "Path ${PATH_TO_CHECK} does not exist")
    endif()
endfunction()

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()
if (${TARGET_ARCH} STREQUAL "pcie")
    message( "${TARGET_ARCH} mode, starting......")
    set(lib_DIR /usr/lib/x84_64-linux-gnu)
    link_directories(${lib_DIR})
   
    include_directories(libtorch_tokenizer/src/unilib)
    include_directories(libtorch_tokenizer/src)
    # 添加libsophon的头文件目录
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    # 添加sophon-ffmpeg的头文件及动态库目录
    set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})

    # 添加sophon-opencv的头文件目录
    set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    # 添加SAIL的头文件及动态库目录
    include_directories(/opt/sophon/sophon-sail/include/)
    include_directories(/opt/sophon/sophon-sail/include/sail)
    link_directories(/opt/sophon/sophon-sail/lib)
    
    add_executable(bert_sail.pcie libtorch_tokenizer/src/tokenizer.cpp libtorch_tokenizer/src/unilib/unicode.cpp libtorch_tokenizer/src/unilib/uninorms.cpp bert_sail.cpp bert_sail.hpp utils.hpp main.cpp)
    target_link_libraries(bert_sail.pcie sail ${FFMPEG_LIBS} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} )

    target_link_libraries(bert_sail.pcie ${TORCH_LIBRARIES})
    
elseif (${TARGET_ARCH} STREQUAL "soc")
    message( "${TARGET_ARCH} mode, starting......")
    check_and_replace_path(SDK)
    check_and_replace_path(SAIL_PATH)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    
    include_directories(libtorch_tokenizer/src/unilib)
    include_directories(libtorch_tokenizer/src)
    set(BM_LIBS bmlib bmrt bmcv)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    

    link_directories("${SDK}/lib/")
    # use sail headers
    include_directories(${SAIL_PATH}/include)
    include_directories(${SAIL_PATH}/include/sail)
    # use sail
    set(SAIL_LIBS  ${SAIL_PATH}/lib/libsail.so)
    # sail为需要链接的库
    message("SDK: " ${SDK})

    add_executable(bert_sail.soc libtorch_tokenizer/src/tokenizer.cpp libtorch_tokenizer/src/unilib/unicode.cpp libtorch_tokenizer/src/unilib/uninorms.cpp bert_sail.cpp bert_sail.hpp utils.hpp main.cpp)
    target_link_libraries(bert_sail.soc ${BM_LIBS} ${OPENCV_LIBS}  ${JPU_LIBS} ${SAIL_LIBS})
  
    target_link_libraries(bert_sail.soc ${TORCH_LIBRARIES})
else ()
    message( FATAL_ERROR  "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif ()