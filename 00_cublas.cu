#include "common.hpp"
#include "cublas_v2.h"

void launch_gemm(size_t M, size_t N, size_t K, half *A, half *B, half *C, half alpha, half beta)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A,
                 CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    testError(launch_gemm,0);
    perf_measure(launch_gemm);

}