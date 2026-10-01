#===----------------------------------------------------------------------===//
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===//
# Vendored zxing (from opencv_contrib's wechat-qrcode module; Apache-2.0).
#
# Build the whole vendored tree as a single static library. Internal includes
# use file-relative paths ("../precomp.hpp", "qrcode_reader.hpp"); the public
# headers of the QR pipeline include "zxing/..." from the parent directory, so
# the parent is exposed as a public include dir.

cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE ZXING_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(zxing STATIC ${ZXING_SRC})
target_include_directories(zxing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(zxing PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)