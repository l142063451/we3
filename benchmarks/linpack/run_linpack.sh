#!/bin/bash

# LINPACK Baseline Benchmark
# High-Performance Computing benchmark for CPU FLOPS measurement

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== High-Performance LINPACK Benchmark ==="

# Create optimized LINPACK implementation
cat > linpack.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define N 1000

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// LU decomposition with partial pivoting
int lu_decomposition(double **A, int *P, int n) {
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }
    
    for (int k = 0; k < n - 1; k++) {
        // Find pivot
        int pivot_row = k;
        double max_val = fabs(A[k][k]);
        
        for (int i = k + 1; i < n; i++) {
            if (fabs(A[i][k]) > max_val) {
                max_val = fabs(A[i][k]);
                pivot_row = i;
            }
        }
        
        // Swap rows if needed
        if (pivot_row != k) {
            double *temp_row = A[k];
            A[k] = A[pivot_row];
            A[pivot_row] = temp_row;
            
            int temp_p = P[k];
            P[k] = P[pivot_row];
            P[pivot_row] = temp_p;
        }
        
        // Check for singular matrix
        if (fabs(A[k][k]) < 1e-12) {
            return -1; // Singular matrix
        }
        
        // Elimination
        for (int i = k + 1; i < n; i++) {
            A[i][k] = A[i][k] / A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
    }
    
    return 0; // Success
}

// Forward substitution
void forward_substitution(double **L, double *b, double *y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
    }
}

// Back substitution
void back_substitution(double **U, double *y, double *x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }
}

int main() {
    printf("High-Performance LINPACK Benchmark\n");
    printf("Problem size: %dx%d\n", N, N);
    
    // Allocate memory
    double **A = malloc(N * sizeof(double*));
    double **A_orig = malloc(N * sizeof(double*));
    double *b = malloc(N * sizeof(double));
    double *b_orig = malloc(N * sizeof(double));
    double *x = malloc(N * sizeof(double));
    double *y = malloc(N * sizeof(double));
    int *P = malloc(N * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        A[i] = malloc(N * sizeof(double));
        A_orig[i] = malloc(N * sizeof(double));
    }
    
    printf("Generating random matrix and RHS vector...\n");
    
    // Initialize with random values
    srand(12345); // Fixed seed for reproducibility
    for (int i = 0; i < N; i++) {
        b[i] = (double)rand() / RAND_MAX;
        b_orig[i] = b[i];
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            A_orig[i][j] = A[i][j];
        }
    }
    
    printf("Starting LU decomposition...\n");
    double start_time = get_time();
    
    // Solve Ax = b using LU decomposition
    if (lu_decomposition(A, P, N) != 0) {
        printf("Error: Matrix is singular\n");
        return 1;
    }
    
    // Apply permutation to b
    double *b_perm = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        b_perm[i] = b_orig[P[i]];
    }
    
    // Forward substitution (Ly = Pb)
    forward_substitution(A, b_perm, y, N);
    
    // Back substitution (Ux = y)
    back_substitution(A, y, x, N);
    
    double end_time = get_time();
    double elapsed = end_time - start_time;
    
    printf("LU decomposition completed.\n");
    
    // Calculate FLOPS
    // LU decomposition: (2/3) * n^3 + O(n^2) operations
    // Forward/back substitution: 2 * n^2 operations
    double total_ops = (2.0/3.0) * N * N * N + 2.0 * N * N;
    double gflops = total_ops / elapsed / 1e9;
    
    // Verification - compute residual ||Ax - b||
    double residual = 0.0;
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A_orig[i][j] * x[j];
        }
        double diff = sum - b_orig[i];
        residual += diff * diff;
    }
    residual = sqrt(residual);
    
    printf("\n=== LINPACK Results ===\n");
    printf("Problem size:     %d x %d\n", N, N);
    printf("Time:             %.6f seconds\n", elapsed);
    printf("Total operations: %.0f\n", total_ops);
    printf("Performance:      %.2f GFLOPS\n", gflops);
    printf("Residual norm:    %.2e\n", residual);
    printf("Solution x[0]:    %.6f\n", x[0]);
    
    // Save results to JSON
    FILE *f = fopen("linpack_result.json", "w");
    fprintf(f, "{\n");
    fprintf(f, "  \"test_name\": \"linpack_hpl\",\n");
    fprintf(f, "  \"problem_size\": %d,\n", N);
    fprintf(f, "  \"elapsed_seconds\": %.6f,\n", elapsed);
    fprintf(f, "  \"total_operations\": %.0f,\n", total_ops);
    fprintf(f, "  \"gflops\": %.2f,\n", gflops);
    fprintf(f, "  \"residual_norm\": %.2e,\n", residual);
    fprintf(f, "  \"verification_passed\": %s\n", (residual < 1e-8) ? "true" : "false");
    fprintf(f, "}\n");
    fclose(f);
    
    printf("Results saved to linpack_result.json\n");
    
    // Cleanup
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(A_orig[i]);
    }
    free(A);
    free(A_orig);
    free(b);
    free(b_orig);
    free(b_perm);
    free(x);
    free(y);
    free(P);
    
    return 0;
}
EOF

echo "Compiling LINPACK benchmark..."
gcc -O3 -march=native -o linpack linpack.c -lm

echo "Running LINPACK benchmark..."
./linpack

echo "LINPACK benchmark completed."