/*
 * CUDA Merge Sort Implementation 
 * GPU Parallel Merge (No CPU Bottleneck)
 * 
 * Compile:
 *   nvcc -O3 -arch=sm_75 -std=c++11 -o cuda_sort cuda.cu
 *   (Replace sm_75 with your GPU compute capability)
 * 
 * Run:
 *   ./cuda_sort N THREADS_PER_BLOCK
 *   Example: ./cuda_sort 10000000 256
 * 
 * Default: N=10000000, THREADS_PER_BLOCK=256
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

// Portable random number generator
static uint64_t rng_state = 123456;

void seed_rng(uint64_t seed) {
    rng_state = seed;
}

int portable_rand() {
    rng_state = (rng_state * 6364136223846793005ULL + 1442695040888963407ULL);
    return (int)((rng_state >> 32) & 0x7FFFFFFF);
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU kernel: Bitonic sort for chunks within a block
__global__ void bitonicSortKernel(int *arr, int n, int chunk_size) {
    extern __shared__ int shared_data[];
    
    int chunk_id = blockIdx.x;
    int start = chunk_id * chunk_size;
    int tid = threadIdx.x;
    int actual_size = min(chunk_size, n - start);
    
    if (start >= n) return;
    
    // Find next power of 2
    int size_pow2 = 1;
    while (size_pow2 < actual_size) size_pow2 *= 2;
    
    // Load data into shared memory
    if (tid < actual_size) {
        shared_data[tid] = arr[start + tid];
    } else if (tid < size_pow2) {
        shared_data[tid] = 2147483647; // INT_MAX for padding
    }
    __syncthreads();
    
    // Bitonic sort
    for (int k = 2; k <= size_pow2; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            
            if (ixj > tid && tid < size_pow2 && ixj < size_pow2) {
                int ascending = ((tid & k) == 0);
                
                if ((ascending && shared_data[tid] > shared_data[ixj]) || 
                    (!ascending && shared_data[tid] < shared_data[ixj])) {
                    // Swap
                    int temp = shared_data[tid];
                    shared_data[tid] = shared_data[ixj];
                    shared_data[ixj] = temp;
                }
            }
            __syncthreads();
        }
    }
    
    // Write back only valid elements
    if (tid < actual_size) {
        arr[start + tid] = shared_data[tid];
    }
}

// GPU kernel: Parallel merge of sorted chunks
__global__ void parallelMergeKernel(int *src, int *dst, int n, int chunk_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;
    
    // Determine which pair of chunks this thread belongs to
    int merge_id = tid / (2 * chunk_size);
    int local_id = tid % (2 * chunk_size);
    
    int left_start = merge_id * 2 * chunk_size;
    int mid = left_start + chunk_size;
    int right_end = min(left_start + 2 * chunk_size, n);
    
    // Boundary check
    if (left_start >= n) return;
    mid = min(mid, n);
    
    // Binary search to find position in merged array
    int target_pos = left_start + local_id;
    if (target_pos >= right_end) return;
    
    // Determine which sorted chunk and position
    int left_size = mid - left_start;
    int right_size = right_end - mid;
    
    // Binary search in left chunk for elements <= target
    int left_pos = 0;
    if (local_id < left_size) {
        // This element is from left chunk
        int value = src[left_start + local_id];
        
        // Count elements in right chunk that are smaller
        int l = 0, r = right_size;
        while (l < r) {
            int m = (l + r) / 2;
            if (mid + m < n && src[mid + m] < value) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        left_pos = local_id + l;
        dst[left_start + left_pos] = value;
    } else if (local_id < left_size + right_size) {
        // This element is from right chunk
        int right_idx = local_id - left_size;
        int value = src[mid + right_idx];
        
        // Count elements in left chunk that are <= value
        int l = 0, r = left_size;
        while (l < r) {
            int m = (l + r) / 2;
            if (left_start + m < n && src[left_start + m] <= value) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        left_pos = l + right_idx;
        dst[left_start + left_pos] = value;
    }
}

// Verify and checksum
int verify_and_checksum(int *arr, int n, uint64_t *checksum) {
    *checksum = 0;
    for (int i = 0; i < n; i++) {
        *checksum += (uint64_t)arr[i];
        if (i > 0 && arr[i] < arr[i-1]) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int N = 10000000;
    int threads_per_block = 256;
    
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid N: %s\n", argv[1]);
            return 1;
        }
    }
    
    if (argc > 2) {
        threads_per_block = atoi(argv[2]);
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            fprintf(stderr, "Invalid threads_per_block (1-1024): %s\n", argv[2]);
            return 1;
        }
        // Ensure power of 2
        int pow2 = 1;
        while (pow2 < threads_per_block) pow2 *= 2;
        if (pow2 != threads_per_block) {
            fprintf(stderr, "threads_per_block must be power of 2 (64, 128, 256, 512, 1024)\n");
            return 1;
        }
    }
    
    int chunk_size = threads_per_block;
    int num_blocks = (N + chunk_size - 1) / chunk_size;
    
    // Allocate host memory
    int *h_arr = (int*)malloc(N * sizeof(int));
    
    if (!h_arr) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }
    
    // Fill array
    seed_rng(123456);
    for (int i = 0; i < N; i++) {
        h_arr[i] = portable_rand();
    }
    
    // Allocate device memory (need two buffers for ping-pong merging)
    int *d_arr, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_temp, N * sizeof(int)));
    
    // Events
    cudaEvent_t start_h2d, end_h2d, start_sort, end_sort, start_merge, end_merge, start_d2h, end_d2h;
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&end_h2d));
    CUDA_CHECK(cudaEventCreate(&start_sort));
    CUDA_CHECK(cudaEventCreate(&end_sort));
    CUDA_CHECK(cudaEventCreate(&start_merge));
    CUDA_CHECK(cudaEventCreate(&end_merge));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&end_d2h));
    
    // H2D
    CUDA_CHECK(cudaEventRecord(start_h2d));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(end_h2d));
    
    // Sort phase: Sort chunks on GPU
    size_t shared_mem_size = threads_per_block * sizeof(int);
    CUDA_CHECK(cudaEventRecord(start_sort));
    bitonicSortKernel<<<num_blocks, threads_per_block, shared_mem_size>>>(d_arr, N, chunk_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end_sort));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Merge phase: Parallel merge on GPU
    CUDA_CHECK(cudaEventRecord(start_merge));
    
    int current_chunk = chunk_size;
    int *src = d_arr;
    int *dst = d_temp;
    
    while (current_chunk < N) {
        // Calculate threads needed for merge
        int merge_threads = 256;
        int merge_blocks = (N + merge_threads - 1) / merge_threads;
        
        parallelMergeKernel<<<merge_blocks, merge_threads>>>(src, dst, N, current_chunk);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Swap buffers (ping-pong)
        int *temp = src;
        src = dst;
        dst = temp;
        
        current_chunk *= 2;
    }
    
    CUDA_CHECK(cudaEventRecord(end_merge));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Final result is in src
    int *final_arr = src;
    
    // D2H
    CUDA_CHECK(cudaEventRecord(start_d2h));
    CUDA_CHECK(cudaMemcpy(h_arr, final_arr, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(end_d2h));
    CUDA_CHECK(cudaEventSynchronize(end_d2h));
    
    // Timing
    float time_h2d, time_sort, time_merge, time_d2h;
    CUDA_CHECK(cudaEventElapsedTime(&time_h2d, start_h2d, end_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&time_sort, start_sort, end_sort));
    CUDA_CHECK(cudaEventElapsedTime(&time_merge, start_merge, end_merge));
    CUDA_CHECK(cudaEventElapsedTime(&time_d2h, start_d2h, end_d2h));
    
    float gpu_time_ms = time_sort + time_merge;
    float total_time_ms = time_h2d + time_sort + time_merge + time_d2h;
    
    // Verify
    uint64_t checksum;
    int sorted = verify_and_checksum(h_arr, N, &checksum);
    
    // Print
    printf("CUDA N=%d config=threads_%d_blocks_%d time_ms=%.2f total_time_ms=%.2f correctness=%s checksum=%lu\n",
           N, threads_per_block, num_blocks, gpu_time_ms, total_time_ms, 
           sorted ? "OK" : "FAIL", checksum);
    printf("  Breakdown: H2D=%.2fms Sort=%.2fms Merge=%.2fms D2H=%.2fms\n",
           time_h2d, time_sort, time_merge, time_d2h);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start_h2d));
    CUDA_CHECK(cudaEventDestroy(end_h2d));
    CUDA_CHECK(cudaEventDestroy(start_sort));
    CUDA_CHECK(cudaEventDestroy(end_sort));
    CUDA_CHECK(cudaEventDestroy(start_merge));
    CUDA_CHECK(cudaEventDestroy(end_merge));
    CUDA_CHECK(cudaEventDestroy(start_d2h));
    CUDA_CHECK(cudaEventDestroy(end_d2h));
    CUDA_CHECK(cudaFree(d_arr));
    CUDA_CHECK(cudaFree(d_temp));
    free(h_arr);
    
    return 0;
}
