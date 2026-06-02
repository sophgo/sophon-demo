cmake_minimum_required(VERSION 3.10)
project(seaco_paraformer_bmrt)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_STANDARD 14)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

if (${TARGET_ARCH} STREQUAL "pcie")
    message("${TARGET_ARCH} mode, starting......")

    # libsophon (bmlib, bmrt, bmcv)
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    # Armadillo
    find_package(Armadillo REQUIRED)
    include_directories(${ARMADILLO_INCLUDE_DIRS})

    # libsndfile
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SNDFILE REQUIRED sndfile)
    include_directories(${SNDFILE_INCLUDE_DIRS})

    aux_source_directory(. SRC_FILES)
    add_executable(seaco_paraformer_bmrt.pcie ${SRC_FILES})
    target_link_libraries(seaco_paraformer_bmrt.pcie
        ${ARMADILLO_LIBRARIES}
        ${SNDFILE_LIBRARIES}
        ${the_libbmlib.so}
        ${the_libbmrt.so}
        ${the_libbmcv.so}
        -lpthread
    )
    target_include_directories(seaco_paraformer_bmrt.pcie PRIVATE ${SNDFILE_INCLUDE_DIRS})

elseif (${TARGET_ARCH} STREQUAL "soc")
    message("${TARGET_ARCH} mode, starting......")

    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

    set(BM_LIBS bmlib bmrt bmcv)
    set(SDK_DIR "${SDK}")

    include_directories("${SDK_DIR}/include/")
    include_directories("${SDK_DIR}/include/opencv4")
    include_directories("${PROJECT_SOURCE_DIR}/cross_compile_module/3rd_party/include")
    link_directories("${SDK_DIR}/lib/")
    link_directories("${PROJECT_SOURCE_DIR}/cross_compile_module/3rd_party/lib")

    set(ARMADILLO_LIBRARIES ${PROJECT_SOURCE_DIR}/cross_compile_module/3rd_party/lib/libarmadillo.so)
    set(SNDFILE_LIBRARIES ${PROJECT_SOURCE_DIR}/cross_compile_module/3rd_party/lib/libsndfile.so)
    set(BLAS_LIB ${PROJECT_SOURCE_DIR}/cross_compile_module/3rd_party/lib/blas/libblas.so.3)
    set(LAPACK_LIB ${PROJECT_SOURCE_DIR}/cross_compile_module/3rd_party/lib/lapack/liblapack.so.3)

    message("SDK: ${SDK_DIR}")

    aux_source_directory(. SRC_FILES)
    add_executable(seaco_paraformer_bmrt.soc ${SRC_FILES})
    target_link_libraries(seaco_paraformer_bmrt.soc
        ${ARMADILLO_LIBRARIES}
        ${SNDFILE_LIBRARIES}
        ${BLAS_LIB}
        ${LAPACK_LIB}
        ${BM_LIBS}
        -lpthread
    )

else()
    message(FATAL_ERROR "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif()
