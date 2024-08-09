#include "common.hpp"
using namespace nvcuda;

#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

#define WARP_SIZE 32


__global__ void naiveBlockKernel(const half *A, const half *B, half *C,
                                 size_t M, size_t N, size_t K)
{
    const size_t K_tiles = CEIL_DIV(K, BLOCK_K);

    const size_t c_row = blockIdx.y * BLOCK_M;
    const size_t c_col = blockIdx.x * BLOCK_N;

    if (c_row >= M && c_col >= N)
    {
        return;
    }

    wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i)
    {
        wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::col_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + c_row * K + i * BLOCK_K, K);
        wmma::load_matrix_sync(B_frag, B + i * BLOCK_K + c_col * K, K);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }
    wmma::store_matrix_sync(C + c_row * N + c_col, C_frag, N, wmma::mem_row_major);
}

void launch_gemm(size_t M, size_t N, size_t K, half *A, half *B, half *C, half alpha,half beta)
{
    dim3 block(WARP_SIZE);
    dim3 grid(CEIL_DIV(N, BLOCK_N), CEIL_DIV(M, BLOCK_M));

    naiveBlockKernel<<<grid, block>>>(A, B, C, M, N, K);
}


int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    testError(launch_gemm,0);
    perf_measure(launch_gemm);
}