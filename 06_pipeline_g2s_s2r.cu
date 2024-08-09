#include "common.hpp"
using namespace nvcuda;

// BlockTile的Shape
#define BT_M 256
#define BT_N 128

// WMMA-TensorCore执行计算的Shape
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16

// BlockTile内按照Warp 2x4拆分
#define BT_ROW_WT_NUM 2 // BlockTile每一行分为2个WarpTile
#define BT_COL_WT_NUM 4 // BlockTile每一列分为4个WarpTile

// WarpTile的Shape
#define WT_M (BT_M / BT_COL_WT_NUM) // WarpTile M-Axis的元素个数
#define WT_N (BT_N / BT_ROW_WT_NUM) // WarpTile N-Axis的元素个数

// 每个BlockTile的MMA Tile的数量
#define BT_COL_MMA_NUM (BT_M / MMA_M) // BlockTile每一列包含的MMA_TILE的数量
#define BT_ROW_MMA_NUM (BT_N / MMA_N) // BlockTile每一行包含的MMA_TILE的数量

// 每个WarpTile的MMA Tile的数量
#define WT_COL_MMA_NUM (WT_M / MMA_M) // WarpTile每一列包含MMA_TILE的数量
#define WT_ROW_MMA_NUM (WT_N / MMA_N) // WarpTile每一行包含MMA_TILE的数量

// 一个WARP有32个线程, 一个BlockTile内的线程数为BT_THREAD_NUM
#define WARP_SIZE 32
#define BT_WARP_NUM (BT_ROW_WT_NUM * BT_COL_WT_NUM)
#define BT_THREAD_NUM (WARP_SIZE * BT_WARP_NUM)

#define CHUNK_K 2      // 每次处理的MMA_TILE_K的Batch个数
#define SKEW_PADDING 8 // 为了解决BankConflict增加的Padding
#define MMA_SMEM_STRIDE_K (CHUNK_K * MMA_K + SKEW_PADDING)
#define C_SMEM_STRIDE (BT_N + SKEW_PADDING)

#define CHUNK_LINE_BYTES (CHUNK_K * MMA_K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define THREAD_COPY_BYTES 16

__global__ void blockGemmKernel(half *A, half *B, half *C, size_t M, size_t N, size_t K)
{
    // 矩阵被分块成MMA_Tile的各维度个数
    const size_t M_tiles = CEIL_DIV(M, MMA_N);
    const size_t N_tiles = CEIL_DIV(N, MMA_M);
    const size_t K_tiles = CEIL_DIV(K, MMA_K);

    // 根据blockIdx查找计算的MMA_TILE的坐标
    const size_t block_tile_i = blockIdx.x * BT_COL_MMA_NUM;
    const size_t block_tile_j = blockIdx.y * BT_ROW_MMA_NUM;

    // OOB(Out-Of-bound)判断
    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles)
    {
        return;
    }

    extern __shared__ half shmem[][MMA_SMEM_STRIDE_K];

    // warp_id和lane_id定义,对齐PTX相关的文档
    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    // 基于MMA_TILE在WARP_LEVEL初始化C_fragment数组
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, half> C_frag[WT_COL_MMA_NUM][WT_ROW_MMA_NUM];
#pragma unroll
    for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)
    {
#pragma unroll
        for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)
        {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }
    // B为Col-major存储, 因此Offset为Y轴的元素个数BT_M
    constexpr size_t shmem_idx_b_off = BT_M;
    constexpr size_t shmem_cache_off = BT_M + BT_N;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    half *shmem_warp_tile_ptr = &shmem[0][0] +
                                (warp_id / BT_ROW_WT_NUM) * C_SMEM_STRIDE * WT_M +
                                (warp_id % BT_ROW_WT_NUM) * WT_N;

    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory
    half *shmem_warp_stream_ptr = &shmem[0][0] + warp_id * MMA_M * 2 * C_SMEM_STRIDE;

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx =
        (block_tile_i + warp_id * 2) * MMA_M * N + block_tile_j * MMA_N;

    half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // 加载AB矩阵的GMEM指针
    const half *A_warp_ptr = &A[block_tile_i * MMA_M * K] + BT_M / BT_WARP_NUM * K * warp_id;
    const half *B_warp_ptr = &B[block_tile_j * MMA_N * K] + BT_N / BT_WARP_NUM * K * warp_id;

    // 每次迭代的拷贝数据量
    constexpr size_t A_smem_iters = BT_M / (CHUNK_COPY_LINES_PER_WARP * BT_WARP_NUM);
    constexpr size_t B_smem_iters = BT_N / (CHUNK_COPY_LINES_PER_WARP * BT_WARP_NUM);

    size_t shmem_store_off = 0;
    size_t shmem_load_off = shmem_cache_off;

    // 将A矩阵的Chunk从GMEM拷贝到SMEM
    size_t A_smem_idx = shmem_store_off + BT_M / BT_WARP_NUM * warp_id;
    int4 *A_lane_ptr = (int4 *)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i)
    {
        uint32_t A_smem_lane_addr =
            __cvta_generic_to_shared(&shmem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    // 将B矩阵的Chunk从GMEM拷贝到SMEM
    size_t B_smem_idx = shmem_store_off + shmem_idx_b_off + BT_N / BT_WARP_NUM * warp_id;
    int4 *B_lane_ptr = (int4 *)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i)
    {
        uint32_t B_smem_lane_addr =
            __cvta_generic_to_shared(&shmem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);

    // 同步等待完成shmem_store_off buffer拷贝
    __syncthreads();

// Loop for Block_Tile_K
#pragma unroll
    for (size_t tile_k = CHUNK_K; tile_k < K_tiles; tile_k += CHUNK_K)
    {
        //循环中交替使用buffer的Offset
        shmem_store_off ^= shmem_cache_off;
        shmem_load_off ^= shmem_cache_off;

        A_smem_idx = shmem_store_off + BT_M / BT_WARP_NUM * warp_id;
        A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_smem_iters; ++i)
        {
            uint32_t A_smem_lane_addr =
                __cvta_generic_to_shared(&shmem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        B_smem_idx = shmem_store_off + shmem_idx_b_off + BT_N / BT_WARP_NUM * warp_id;
        B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_smem_iters; ++i)
        {
            uint32_t B_smem_lane_addr =
                __cvta_generic_to_shared(&shmem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        // 构建2个Fragment, 进行替换
        wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major>A_frag[2][WT_COL_MMA_NUM];
        wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::col_major>B_frag[2][WT_ROW_MMA_NUM];
        size_t reg_store_idx = 0;
        size_t reg_load_idx = 1;

// 循环前LOAD-A
#pragma unroll
        for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)
        {
            size_t A_smem_idx = shmem_load_off + (warp_id / BT_ROW_WT_NUM) * WT_M + i * MMA_M;
            const half *A_tile_ptr = &shmem[A_smem_idx][0];

            wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, MMA_SMEM_STRIDE_K);
        }
//循环前LOAD-B
#pragma unroll
        for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)
        {
            size_t B_smem_idx = shmem_load_off + shmem_idx_b_off + (warp_id % BT_ROW_WT_NUM) * WT_N + j * MMA_N;
            const half *B_tile_ptr = &shmem[B_smem_idx][0];
            wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, MMA_SMEM_STRIDE_K);
        }

// WarpTile计算GEMM, 对加载的CHUNK处理
#pragma unroll
        for (size_t k_step = 1; k_step < CHUNK_K; ++k_step)
        {
            reg_store_idx ^= 1;
            reg_load_idx ^= 1;

            // 将A-Fragment从SMEM移动到寄存器
#pragma unroll
            for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)
            {
                size_t A_smem_idx = shmem_load_off + (warp_id / BT_ROW_WT_NUM) * WT_M + i * MMA_M;
                const half *A_tile_ptr = &shmem[A_smem_idx][k_step * MMA_K];

                wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, MMA_SMEM_STRIDE_K);
            }
            // 将B-Fragment从SMEM移动到寄存器
#pragma unroll
            for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)
            {
                size_t B_smem_idx = shmem_load_off + shmem_idx_b_off + (warp_id % BT_ROW_WT_NUM) * WT_N + j * MMA_N;
                const half *B_tile_ptr = &shmem[B_smem_idx][k_step * MMA_K];
                wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, MMA_SMEM_STRIDE_K);
            }

            // 执行TensorCore MMA计算
#pragma unroll
            for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)
            {
#pragma unroll
                for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)
                {
                    wmma::mma_sync(C_frag[i][j], A_frag[reg_load_idx][i], B_frag[reg_load_idx][j], C_frag[i][j]);
                }
            }
        }

#pragma unroll
            for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)
            {
#pragma unroll
                for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)
                {
                    wmma::mma_sync(C_frag[i][j], A_frag[reg_store_idx][i], B_frag[reg_store_idx][j], C_frag[i][j]);
                }
            }        

        // 计算和异步拷贝Overlap
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
        // 完成GEMM计算并同步
        __syncthreads();
    }

    // 基于最后的shmem_store_off计算MMA

    // 构建2个Fragment, 进行替换
    wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> A_frag[2][WT_COL_MMA_NUM];
    wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::col_major> B_frag[2][WT_ROW_MMA_NUM];
    size_t reg_store_idx = 0;
    size_t reg_load_idx = 1;

#pragma unroll
    for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)
    {
        size_t A_smem_idx = shmem_store_off + (warp_id / BT_ROW_WT_NUM) * WT_M + i * MMA_M;
        const half *A_tile_ptr = &shmem[A_smem_idx][0];

        wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, MMA_SMEM_STRIDE_K);
    }
#pragma unroll
    for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)
    {
        size_t B_smem_idx = shmem_store_off + shmem_idx_b_off + (warp_id % BT_ROW_WT_NUM) * WT_N + j * MMA_N;
        const half *B_tile_ptr = &shmem[B_smem_idx][0];
        wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, MMA_SMEM_STRIDE_K);
    }

#pragma unroll
    for (size_t k_step = 1; k_step < CHUNK_K; ++k_step)
    {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;
        // 将A-Fragment从SMEM移动到寄存器
#pragma unroll
        for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)
        {
            size_t A_smem_idx = shmem_store_off + (warp_id / BT_ROW_WT_NUM) * WT_M + i * MMA_M;
            const half *A_tile_ptr = &shmem[A_smem_idx][k_step * MMA_K];

            wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, MMA_SMEM_STRIDE_K);
        }
        // 将B-Fragment从SMEM移动到寄存器
#pragma unroll
        for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)
        {
            size_t B_smem_idx = shmem_store_off + shmem_idx_b_off + (warp_id % BT_ROW_WT_NUM) * WT_N + j * MMA_N;
            const half *B_tile_ptr = &shmem[B_smem_idx][k_step * MMA_K];
            wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, MMA_SMEM_STRIDE_K);
        }

        // 执行TensorCore MMA计算
#pragma unroll
        for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)
        {
#pragma unroll
            for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)
            {
                wmma::mma_sync(C_frag[i][j], A_frag[reg_load_idx][i], B_frag[reg_load_idx][j], C_frag[i][j]);
            }
        }
    }

    // last RF MMA
#pragma unroll
    for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)
    {
#pragma unroll
        for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)
        {
            wmma::mma_sync(C_frag[i][j], A_frag[reg_store_idx][i], B_frag[reg_store_idx][j], C_frag[i][j]);
        }
    }

    // 完成shmem_store_off计算MMA同步
    __syncthreads();

    // WMMA-STORE 保存结果C矩阵到SHMEM
#pragma unroll
    for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)
    {
#pragma unroll
        for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)
        {
            half *C_tile_ptr = shmem_warp_tile_ptr + i * C_SMEM_STRIDE * MMA_M + j * MMA_N;
            wmma::store_matrix_sync(C_tile_ptr, C_frag[i][j], C_SMEM_STRIDE, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // 对齐写回到GMEM
#pragma unroll
    for (size_t i = 0; i < MMA_M; ++i)
    {
        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(shmem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) + lane_id % 16);
    }
}

void launch_gemm(size_t M, size_t N, size_t K, half *A, half *B, half *C, half alpha, half beta)
{
    // 获取平台SHMEM SIZE
    int dev_id = 0;
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, dev_id);

    size_t SHMEM_SZ =
        std::max((BT_M + BT_N) * MMA_SMEM_STRIDE_K * sizeof(half) * 2, BT_M * C_SMEM_STRIDE * sizeof(half));

    if (dev_prop.sharedMemPerMultiprocessor > SHMEM_SZ)
        cudaFuncSetAttribute(blockGemmKernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SHMEM_SZ);

    dim3 block(BT_THREAD_NUM);
    dim3 grid(CEIL_DIV(M, BT_M), CEIL_DIV(N, BT_N));
    blockGemmKernel<<<grid, block, SHMEM_SZ>>>(A, B, C, M, N, K);
}

int main()
{
    testError(launch_gemm, 0);
    perf_measure(launch_gemm);
}