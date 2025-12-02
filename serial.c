/*
 * Serial Merge Sort Implementation
 * 
 * Compile:
 *   gcc -O3 -march=native -std=c11 -o serial serial.c
 * 
 * Run:
 *   ./serial N
 *   Example: ./serial 10000000
 * 
 * Default N = 10000000 if not provided
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

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

// Recursive merge sort
void merge_sort(int *arr, int *tmp, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, tmp, left, mid);
        merge_sort(arr, tmp, mid + 1, right);
        merge(arr, tmp, left, mid, right);
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
    srand(123456);
    for (int i = 0; i < N; i++) {
        arr[i] = rand();
    }
    
    // Start timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Sort
    merge_sort(arr, tmp, 0, N - 1);
    
    // End timing
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    // Verify and compute checksum
    uint64_t checksum;
    int sorted = verify_and_checksum(arr, N, &checksum);
    
    // Print result
    printf("Serial N=%d config=none time_ms=%.2f correctness=%s checksum=%lu\n",
           N, time_ms, sorted ? "OK" : "FAIL", checksum);
    
    // Cleanup
    free(arr);
    free(tmp);
    
    return 0;
}
