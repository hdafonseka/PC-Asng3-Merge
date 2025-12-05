/*
 * OpenMP Parallel Merge Sort Implementation
 * 
 * Compile:
 *   gcc -O3 -fopenmp -march=native -std=c11 -o openmp openmp.c
 * 
 * Run:
 *   OMP_NUM_THREADS=8 ./openmp N
 *   Example: OMP_NUM_THREADS=8 ./openmp 10000000
 *   Or: ./openmp N (uses default thread count)
 * 
 * Default N = 10000000 if not provided
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <omp.h>

// Portable random number generator (LCG) - same on all systems
static uint64_t rng_state = 123456;

void seed_rng(uint64_t seed) {
    rng_state = seed;
}

int portable_rand() {
    rng_state = (rng_state * 6364136223846793005ULL + 1442695040888963407ULL);
    return (int)((rng_state >> 32) & 0x7FFFFFFF);
}

// Threshold for switching from parallel to sequential
#define SEQ_THRESHOLD (1 << 14)  // 16384

// Merge two sorted subarrays [left..mid] and [mid+1..right]
void merge(int *arr, int *tmp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            tmp[k++] = arr[i++];
        } else {
            tmp[k++] = arr[j++];
        }
    }
    
    while (i <= mid) {
        tmp[k++] = arr[i++];
    }
    
    while (j <= right) {
        tmp[k++] = arr[j++];
    }
    
    // Copy back to original array
    for (i = left; i <= right; i++) {
        arr[i] = tmp[i];
    }
}

// Sequential merge sort (used below threshold)
void merge_sort_seq(int *arr, int *tmp, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort_seq(arr, tmp, left, mid);
        merge_sort_seq(arr, tmp, mid + 1, right);
        merge(arr, tmp, left, mid, right);
    }
}

// Parallel merge sort using OpenMP tasks
void merge_sort_parallel(int *arr, int *tmp, int left, int right) {
    if (left < right) {
        int len = right - left + 1;
        
        // Use sequential sort for small segments
        if (len <= SEQ_THRESHOLD) {
            merge_sort_seq(arr, tmp, left, right);
        } else {
            int mid = left + (right - left) / 2;
            
            // Spawn tasks for left and right halves
            #pragma omp task shared(arr, tmp)
            merge_sort_parallel(arr, tmp, left, mid);
            
            #pragma omp task shared(arr, tmp)
            merge_sort_parallel(arr, tmp, mid + 1, right);
            
            // Wait for both tasks to complete
            #pragma omp taskwait
            
            // Merge the sorted halves
            merge(arr, tmp, left, mid, right);
        }
    }
}

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
    
    // Allocate arrays
    int *arr = (int*)malloc(N * sizeof(int));
    int *tmp = (int*)malloc(N * sizeof(int));
    
    if (!arr || !tmp) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Fill array deterministically
    seed_rng(123456);
    for (int i = 0; i < N; i++) {
        arr[i] = portable_rand();
    }
    
    int num_threads;
    
    // Start timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Sort using OpenMP parallel region with tasks
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            merge_sort_parallel(arr, tmp, 0, N - 1);
        }
    }
    
    // End timing
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    // Verify and compute checksum
    uint64_t checksum;
    int sorted = verify_and_checksum(arr, N, &checksum);
    
    // Print result
    printf("OpenMP N=%d config=threads:%d time_ms=%.2f correctness=%s checksum=%lu\n",
           N, num_threads, time_ms, sorted ? "OK" : "FAIL", checksum);
    
    // Cleanup
    free(arr);
    free(tmp);
    
    return 0;
}
