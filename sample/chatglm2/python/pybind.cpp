//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../cpp/chatglm2.hpp"
#include <pybind11/pybind11.h>


PYBIND11_MODULE(ChatGLM2, m) {
    pybind11::class_<ChatGLM2>(m, "ChatGLM2")
        .def(pybind11::init<>())
        .def("init", &ChatGLM2::init)
        .def("chat", &ChatGLM2::chat)
        .def("answer", &ChatGLM2::answer)
        .def("deinit", &ChatGLM2::deinit);
}
