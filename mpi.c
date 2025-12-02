/*
 * MPI Parallel Merge Sort Implementation
 * 
 * Compile:
 *   mpicc -O3 -march=native -std=c11 -o mpi_sort mpi.c
 * 
 * Run:
 *   mpirun -np 4 ./mpi_sort N
 *   Example: mpirun -np 4 ./mpi_sort 10000000
 * 
 * Default N = 10000000 if not provided
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>

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

// Sequential merge sort
void merge_sort(int *arr, int *tmp, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, tmp, left, mid);
        merge_sort(arr, tmp, mid + 1, right);
        merge(arr, tmp, left, mid, right);
    }
}

// Merge two sorted arrays into a result array
void merge_two_arrays(int *arr1, int n1, int *arr2, int n2, int *result) {
    int i = 0, j = 0, k = 0;
    
    while (i < n1 && j < n2) {
        if (arr1[i] <= arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }
    
    while (i < n1) {
        result[k++] = arr1[i++];
    }
    
    while (j < n2) {
        result[k++] = arr2[j++];
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
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            if (rank == 0) {
                fprintf(stderr, "Invalid N: %s\n", argv[1]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    int *arr = NULL;
    int *local_arr = NULL;
    int *tmp = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;
    
    double start_time, end_time;
    
    // Calculate chunk sizes
    int base_chunk = N / size;
    int remainder = N % size;
    int local_n = base_chunk + (rank < remainder ? 1 : 0);
    
    // Allocate local arrays
    local_arr = (int*)malloc(local_n * sizeof(int));
    tmp = (int*)malloc(local_n * sizeof(int));
    
    if (!local_arr || !tmp) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Rank 0 initializes the array
    if (rank == 0) {
        arr = (int*)malloc(N * sizeof(int));
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        if (!arr || !sendcounts || !displs) {
            fprintf(stderr, "Memory allocation failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Fill array deterministically
        srand(123456);
        for (int i = 0; i < N; i++) {
            arr[i] = rand();
        }
        
        // Prepare scatter parameters
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = base_chunk + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
        
        start_time = MPI_Wtime();
    }
    
    // Broadcast sendcounts to all ranks (needed for Scatterv)
    if (rank != 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
    }
    MPI_Bcast(sendcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Scatter data to all ranks
    MPI_Scatterv(arr, sendcounts, displs, MPI_INT,
                 local_arr, local_n, MPI_INT,
                 0, MPI_COMM_WORLD);
    
    // Each rank sorts its local chunk
    merge_sort(local_arr, tmp, 0, local_n - 1);
    
    // Gather sorted chunks back to rank 0
    MPI_Gatherv(local_arr, local_n, MPI_INT,
                arr, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    
    // Rank 0 merges all sorted chunks
    if (rank == 0) {
        // Simple sequential merge of all chunks
        int *temp_arr = (int*)malloc(N * sizeof(int));
        if (!temp_arr) {
            fprintf(stderr, "Memory allocation failed for temp array\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Copy first chunk
        memcpy(temp_arr, arr, sendcounts[0] * sizeof(int));
        int merged_size = sendcounts[0];
        
        // Merge remaining chunks one by one
        for (int i = 1; i < size; i++) {
            int *result = (int*)malloc((merged_size + sendcounts[i]) * sizeof(int));
            merge_two_arrays(temp_arr, merged_size, 
                           arr + displs[i], sendcounts[i], 
                           result);
            free(temp_arr);
            merged_size += sendcounts[i];
            temp_arr = result;
        }
        
        // Copy final result back to arr
        memcpy(arr, temp_arr, N * sizeof(int));
        free(temp_arr);
        
        end_time = MPI_Wtime();
        double time_ms = (end_time - start_time) * 1000.0;
        
        // Verify and compute checksum
        uint64_t checksum;
        int sorted = verify_and_checksum(arr, N, &checksum);
        
        // Print result
        printf("MPI N=%d config=processes:%d time_ms=%.2f correctness=%s checksum=%lu\n",
               N, size, time_ms, sorted ? "OK" : "FAIL", checksum);
        
        free(arr);
        free(sendcounts);
        free(displs);
    } else {
        free(sendcounts);
    }
    
    // Cleanup
    free(local_arr);
    free(tmp);
    
    MPI_Finalize();
    return 0;
}
