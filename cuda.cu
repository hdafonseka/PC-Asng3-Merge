/*
 * CUDA Merge Sort Implementation using Thrust
 * 
 * Compile:
 *   nvcc -O3 -arch=sm_75 -std=c++11 -o cuda_sort cuda.cu
 *   (Replace sm_75 with your GPU compute capability, e.g., sm_60, sm_70, sm_80, sm_86, sm_89)
 * 
 * Run:
 *   ./cuda_sort N
 *   Example: ./cuda_sort 10000000
 * 
 * Default N = 10000000 if not provided
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Verify array is sorted and compute checksum
int verify_and_checksum(int *arr, int n, uint64_t *checksum) {
    *checksum = 0;
    for (int i = 0; i < n; i++) {
        *checksum += (uint64_t)arr[i];
        if (i > 0 && arr[i] < arr[i-1]) {
            return 0; // Not sorted
        }
    }
    return 1; // Sorted OK
}

int main(int argc, char *argv[]) {
    int N = 10000000; // Default
    
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid N: %s\n", argv[1]);
            return 1;
        }
    }
    
    // Allocate host arrays
    int *h_arr = (int*)malloc(N * sizeof(int));
    
    if (!h_arr) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }
    
    // Fill array deterministically
    srand(123456);
    for (int i = 0; i < N; i++) {
        h_arr[i] = rand();
    }
    
    // Allocate device array
    int *d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(int)));
    
    // Create CUDA events for timing
    cudaEvent_t start_copy_h2d, end_copy_h2d;
    cudaEvent_t start_sort, end_sort;
    cudaEvent_t start_copy_d2h, end_copy_d2h;
    
    CUDA_CHECK(cudaEventCreate(&start_copy_h2d));
    CUDA_CHECK(cudaEventCreate(&end_copy_h2d));
    CUDA_CHECK(cudaEventCreate(&start_sort));
    CUDA_CHECK(cudaEventCreate(&end_sort));
    CUDA_CHECK(cudaEventCreate(&start_copy_d2h));
    CUDA_CHECK(cudaEventCreate(&end_copy_d2h));
    
    // Start total timing
    struct timespec total_start, total_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);
    
    // Copy host to device (timed)
    CUDA_CHECK(cudaEventRecord(start_copy_h2d));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(end_copy_h2d));
    
    // Sort using Thrust (timed)
    CUDA_CHECK(cudaEventRecord(start_sort));
    thrust::device_ptr<int> dev_ptr(d_arr);
    thrust::sort(dev_ptr, dev_ptr + N);
    CUDA_CHECK(cudaEventRecord(end_sort));
    
    // Copy device to host (timed)
    CUDA_CHECK(cudaEventRecord(start_copy_d2h));
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(end_copy_d2h));
    
    // Synchronize to ensure all operations complete
    CUDA_CHECK(cudaEventSynchronize(end_copy_d2h));
    
    // End total timing
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    
    // Calculate elapsed times
    float time_h2d_ms, time_sort_ms, time_d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_h2d_ms, start_copy_h2d, end_copy_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&time_sort_ms, start_sort, end_sort));
    CUDA_CHECK(cudaEventElapsedTime(&time_d2h_ms, start_copy_d2h, end_copy_d2h));
    
    float gpu_time_ms = time_sort_ms;
    float total_gpu_time_ms = time_h2d_ms + time_sort_ms + time_d2h_ms;
    
    double total_host_time_ms = (total_end.tv_sec - total_start.tv_sec) * 1000.0 +
                                (total_end.tv_nsec - total_start.tv_nsec) / 1000000.0;
    
    // Verify and compute checksum
    uint64_t checksum;
    int sorted = verify_and_checksum(h_arr, N, &checksum);
    
    // Print result (showing GPU kernel time as main metric)
    printf("CUDA N=%d config=gpu_sort time_ms=%.2f total_time_ms=%.2f correctness=%s checksum=%lu\n",
           N, gpu_time_ms, total_gpu_time_ms, sorted ? "OK" : "FAIL", checksum);
    
    // Additional timing breakdown (optional, for debugging)
    // printf("  H2D: %.2f ms, Sort: %.2f ms, D2H: %.2f ms, Host total: %.2f ms\n",
    //        time_h2d_ms, time_sort_ms, time_d2h_ms, total_host_time_ms);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start_copy_h2d));
    CUDA_CHECK(cudaEventDestroy(end_copy_h2d));
    CUDA_CHECK(cudaEventDestroy(start_sort));
    CUDA_CHECK(cudaEventDestroy(end_sort));
    CUDA_CHECK(cudaEventDestroy(start_copy_d2h));
    CUDA_CHECK(cudaEventDestroy(end_copy_d2h));
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);
    
    return 0;
}
