#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)



#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define MAX(a, b) (a > b ? a : b)

#define M_GLOBAL 4096
#define N_GLOBAL 4096
#define K_GLOBAL 4096
#define ITER 100

void perf_measure(void test_gemm(size_t M, size_t N, size_t K, half *A, half *B, half *C, half alpha, half beta))
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M_GLOBAL * K_GLOBAL * sizeof(half));
    cudaMalloc(&d_b, K_GLOBAL * N_GLOBAL * sizeof(half));
    cudaMalloc(&d_c, M_GLOBAL * N_GLOBAL * sizeof(half));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < ITER; i++)
        test_gemm(M_GLOBAL, N_GLOBAL, K_GLOBAL, d_a, d_b, d_c, alpha, beta);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec;
    cudaEventElapsedTime(&msec, start, end);

    long workload = long(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2 * ITER;
    double avg_Gflops = ((double)workload / 1e9) / (double(msec) / 1e3);
    printf("Average Performance  %10.1lf Gflops\n", avg_Gflops);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__host__ void init_host_matrices(half *a, half *b, half *c, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            a[i * K + j] = (half)(rand() % 3);
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            b[i * K + j] = (half)(rand() % 3);
        }
    }
/*
    for (int t = 0; t < M * N; t++)
    {
        c[t] = (half)(rand() % 3);
    }*/

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            c[i * N + j] = (half)(rand() % 3);
        }
    }
}

__host__ void matMultiplyOnHost(half *A, half *B, half *C, half alpha,
                                half beta, int numARows, int numAColumns,
                                int numBRows, int numBColumns, int numCRows,
                                int numCColumns)
{
    for (int i = 0; i < numCRows; i++)
    {
        for (int j = 0; j < numCColumns; j++)
        {
            half temp = 0.0;
            for (int k = 0; k < numAColumns; k++)
            {
                temp += A[i * numAColumns + k] * B[j * numBRows + k];
            }
            C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
        }
        if (i % 10 == 0)
        {
            printf("CPU Gemm %1.1f%% Completed\n", i * 100.0 / (float)numCRows);
        }
    }
}

int testOnly(void test_gemm(size_t M, size_t N, size_t K, half *A, half *B, half *C, half alpha, half beta))
{

    const int M_TEST = 1024;
    const int N_TEST = 1024;
    const int K_TEST = 512;

    half *A_h = NULL;
    half *B_h = NULL;
    half *C_h = NULL;

    A_h = (half *)malloc(sizeof(half) * M_TEST * K_TEST);
    B_h = (half *)malloc(sizeof(half) * K_TEST * N_TEST);
    C_h = (half *)malloc(sizeof(half) * M_TEST * N_TEST);
    init_host_matrices(A_h, B_h, C_h, M_TEST, N_TEST, K_TEST);

    half *A = NULL;
    half *B = NULL;
    half *C = NULL;

    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(half) * M_TEST * K_TEST);
    cudaMalloc(reinterpret_cast<void **>(&B), sizeof(half) * N_TEST * K_TEST);
    cudaMalloc(reinterpret_cast<void **>(&C), sizeof(half) * M_TEST * N_TEST);
    cudaMemcpy(A, A_h, sizeof(half) * M_TEST * K_TEST, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_TEST * K_TEST, cudaMemcpyHostToDevice);
    cudaMemcpy(C, C_h, sizeof(half) * M_TEST * N_TEST, cudaMemcpyHostToDevice);

    const half alpha = 1.0f;
    const half beta = 0.0f;

    printf("launch cuda gemm kernel\n");
    test_gemm(M_TEST, N_TEST, K_TEST, A, B, C, alpha, beta);

    cudaFree(reinterpret_cast<void *>(A));
    cudaFree(reinterpret_cast<void *>(B));
    cudaFree(reinterpret_cast<void *>(C));
    return 0;
}

int testError(void test_gemm(size_t M, size_t N, size_t K, half *A, half *B, half *C, half alpha, half beta), int scale)
{
    int M_TEST, N_TEST, K_TEST;
    switch (scale)
    {
    case 1:
        M_TEST = 1024;
        N_TEST = 1024;
        K_TEST = 512;
        break;
    case 2:
        M_TEST = 2048;
        N_TEST = 2048;
        K_TEST = 512;
        break;
    case 3:
        M_TEST = 4096;
        N_TEST = 4096;
        K_TEST = 1024;
        break;
    default:
        M_TEST = 512;
        N_TEST = 512;
        K_TEST = 512;
    }

    half *A_h = NULL;
    half *B_h = NULL;
    half *C_h = NULL;
    half *result_hD = NULL;
    half *result_host = NULL;

    A_h = (half *)malloc(sizeof(half) * M_TEST * K_TEST);
    B_h = (half *)malloc(sizeof(half) * K_TEST * N_TEST);
    C_h = (half *)malloc(sizeof(half) * M_TEST * N_TEST);
    result_hD = (half *)malloc(sizeof(half) * M_TEST * N_TEST);
    result_host = (half *)malloc(sizeof(half) * M_TEST * N_TEST);
    init_host_matrices(A_h, B_h, C_h, M_TEST, N_TEST, K_TEST);

    half *A = NULL;
    half *B = NULL;
    half *C = NULL;

    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(half) * M_TEST * K_TEST);
    cudaMalloc(reinterpret_cast<void **>(&B), sizeof(half) * N_TEST * K_TEST);
    cudaMalloc(reinterpret_cast<void **>(&C), sizeof(half) * M_TEST * N_TEST);
    cudaMemcpy(A, A_h, sizeof(half) * M_TEST * K_TEST, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_TEST * K_TEST, cudaMemcpyHostToDevice);
    cudaMemcpy(C, C_h, sizeof(half) * M_TEST * N_TEST, cudaMemcpyHostToDevice);

    const half alpha = 1.0f;
    const half beta = 0.0f;

    printf("launch cuda gemm kernel\n");
    test_gemm(M_TEST, N_TEST, K_TEST, A, B, C, alpha, beta);

    cudaMemcpy(result_hD, C, sizeof(half) * M_TEST * N_TEST, cudaMemcpyDeviceToHost);
    memcpy(result_host, C_h, sizeof(half) * M_TEST * N_TEST);

    printf("launch cpu gemm computation\n");
    matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_TEST, K_TEST,
                      K_TEST, N_TEST, M_TEST, N_TEST);
    printf("starting verify....\n");

    for (int i = 0; i < N_TEST * M_TEST; i++)
    {
        if (fabs((float)result_hD[i] - (float)result_host[i]) > 0.1f)
        {
            printf("mismatch i=%d result_hD=%f result_host=%f\n", i, (float)result_hD[i],
                   (float)result_host[i]);
            break;
        }
    }
    free(result_hD);
    free(result_host);

    free(A_h);
    free(B_h);
    free(C_h);

    cudaFree(reinterpret_cast<void *>(A));
    cudaFree(reinterpret_cast<void *>(B));
    cudaFree(reinterpret_cast<void *>(C));
    return 0;
}
