#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define N 512

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    double **A = malloc(N * sizeof(double*));
    double **B = malloc(N * sizeof(double*));
    double **C = malloc(N * sizeof(double*));
    
    for (int i = 0; i < N; i++) {
        A[i] = malloc(N * sizeof(double));
        B[i] = malloc(N * sizeof(double));
        C[i] = malloc(N * sizeof(double));
    }
    
    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
    }
    
    printf("Starting %dx%d matrix multiplication...\n", N, N);
    
    double start_time = get_time();
    
    // Matrix multiplication
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    double end_time = get_time();
    double elapsed = end_time - start_time;
    
    // Calculate FLOPS (2*N^3 operations for matrix multiply)
    double total_ops = 2.0 * N * N * N;
    double gflops = total_ops / elapsed / 1e9;
    
    printf("Matrix multiplication complete.\n");
    printf("Time: %.6f seconds\n", elapsed);
    printf("Operations: %.0f\n", total_ops);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Result verification: C[0][0] = %.6f\n", C[0][0]);
    
    // Save results
    FILE *f = fopen("cpu_baseline_result.json", "w");
    fprintf(f, "{\n");
    fprintf(f, "  \"test_name\": \"cpu_baseline_dgemm\",\n");
    fprintf(f, "  \"matrix_size\": %d,\n", N);
    fprintf(f, "  \"elapsed_seconds\": %.6f,\n", elapsed);
    fprintf(f, "  \"total_operations\": %.0f,\n", total_ops);
    fprintf(f, "  \"gflops\": %.2f,\n", gflops);
    fprintf(f, "  \"verification_value\": %.6f\n", C[0][0]);
    fprintf(f, "}\n");
    fclose(f);
    
    // Cleanup
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    
    return 0;
}
