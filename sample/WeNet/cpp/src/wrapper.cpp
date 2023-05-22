//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <armadillo>
#include <string>
#include <assert.h>
#include "bmruntime_cpp.h"
#include "bmcv_api.h"
#include "wrapper.h"

using namespace bmruntime;

arma::fmat as_strided(const arma::fmat& X, int n_rows, int n_cols, int row_stride, int col_stride) {
    arma::fmat result(n_rows, n_cols);
    arma::fmat X0 = resize(X, 1, X.n_rows * X.n_cols);
    int start = 0;
    int cur = start;
    for(int i = 0; i < n_rows; i++) {
        cur = start;
        for(int j = 0; j < n_cols; j++) {
            result(i, j) = X(0, cur);
            cur += col_stride;
            cur %= X0.n_cols;
        }
        start += row_stride;
        start %= X0.n_cols;
    } 
    return result;
}

// dim = 0, expand the number of rows; dim = 1, expand the number of columns
arma::fmat pad(const arma::fmat& X, int num, int dim) {
    arma::fmat result(X);
    for(int i = 0; i < num; i++) {
        if(dim == 0) {
            result = arma::join_cols(result, X);
        }
        else {
            result = arma::join_rows(result, X);
        }
    }
    return result;
}

// Generate a matrix of shape [size, 1] with elements increasing linearly from 0 to size-1 
arma::fmat arange(int size) {
    arma::fvec lin_vec = arma::linspace<arma::fvec>(0, size - 1, size);
    arma::fmat lin_mat(size, 1);
    lin_mat.col(0) = lin_vec;
    return lin_mat;
}

// Matrix multiplication, no bugs :)
arma::fmat matmul(const arma::fmat& A, const arma::fmat& B) {
    assert(A.n_cols == B.n_rows && "Matrix multiplication: dimensional mismatch!");
    arma::fmat result(A.n_rows, B.n_cols, arma::fill::zeros);
    for(arma::uword i = 0; i < result.n_rows; i++) {
        for(arma::uword j = 0; j < result.n_cols; j++) {
            for(arma::uword k = 0; k < A.n_cols; k++) {
                result(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return result;
}

// real fft based on bmcv
arma::fmat bm_fft(const arma::fmat& A) {
    int n_rows = A.n_rows;
    int n_cols = A.n_cols;

    /* plan A: There is an error. 2D FFT only supports input sizes that are a power of N where N is 2, 3, 4, or 5. */
    // void* input_real = fmat_to_sys_mem(A);
    // void *out_real_host = std::malloc(n_rows * n_cols * sizeof(float));
    // void *out_imaginary_host = std::malloc(n_rows * n_cols * sizeof(float));

    // bm_handle_t handle = nullptr;
    // bm_dev_request(&handle, 0);

    // bm_device_mem_t input_real_device, out_real_device, out_imaginary_device;
    // bm_malloc_device_byte(handle, &input_real_device, n_rows * n_cols * sizeof(float));
    // bm_malloc_device_byte(handle, &out_real_device, n_rows * n_cols * sizeof(float));
    // bm_malloc_device_byte(handle, &out_imaginary_device, n_rows * n_cols * sizeof(float));
    // bm_memcpy_s2d(handle, input_real_device, input_real);

    // void *plan = nullptr;
    // bmcv_fft_2d_create_plan(handle, n_rows, n_cols, true, plan);
    // bmcv_fft_execute_real_input(handle, input_real_device, out_real_device, out_imaginary_device, plan);

    // bmcv_fft_destroy_plan(handle, plan);
    // bm_memcpy_d2s(handle, out_real_host, out_real_device);
    // bm_memcpy_d2s(handle, out_imaginary_host, out_imaginary_device);
    // bm_free_device(handle, input_real_device);
    // bm_free_device(handle, out_real_device);
    // bm_free_device(handle, out_imaginary_device);
    // bm_dev_free(handle);

    // float* out_real_host_f = static_cast<float*>(out_real_host);
    // arma::fmat out_real_mat(out_real_host_f, n_rows, n_cols);
    // out_real_mat.print("out_real");
    // std::free(input_real);
    // std::free(out_real_host);
    // std::free(out_imaginary_host);
    
    arma::fmat result(n_rows, floor(n_cols / 2) + 1);
    void *out_real_host = std::malloc(n_cols * sizeof(float));
    void *out_imaginary_host = std::malloc(n_cols * sizeof(float));

    bm_handle_t handle = nullptr;
    bm_dev_request(&handle, 0);

    bm_device_mem_t input_real_device, out_real_device, out_imaginary_device;
    bm_malloc_device_byte(handle, &input_real_device, n_cols * sizeof(float));
    bm_malloc_device_byte(handle, &out_real_device, n_cols * sizeof(float));
    bm_malloc_device_byte(handle, &out_imaginary_device, n_cols * sizeof(float));

    void *plan = nullptr;
    bmcv_fft_1d_create_plan(handle, 1, n_cols, true, plan);
    for(int i = 0; i < n_rows; i++) {
        void* input_real = rowvec_to_sys_mem<float>(A.row(i));

        bm_memcpy_s2d(handle, input_real_device, input_real);
        bmcv_fft_execute_real_input(handle, input_real_device, out_real_device, out_imaginary_device, plan);
        bm_memcpy_d2s(handle, out_real_host, out_real_device);
        bm_memcpy_d2s(handle, out_imaginary_host, out_imaginary_device);

        float* out_real_host_f = static_cast<float*>(out_real_host);
        arma::frowvec out_real_frowvec(out_real_host_f, floor(n_cols / 2) + 1);
        float* out_imaginary_host_f = static_cast<float*>(out_imaginary_host);
        arma::frowvec out_imaginary_frowvec(out_imaginary_host_f, floor(n_cols / 2) + 1);
        result.row(i) = arma::pow(arma::pow(out_real_frowvec, 2.0) + arma::pow(out_imaginary_frowvec, 2.0), 0.5);

        std::free(input_real);
    }
    std::free(out_real_host);
    std::free(out_imaginary_host);
    bmcv_fft_destroy_plan(handle, plan);
    bm_free_device(handle, input_real_device);
    bm_free_device(handle, out_real_device);
    bm_free_device(handle, out_imaginary_device);
    bm_dev_free(handle);

    return result;
}

