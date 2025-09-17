#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

#define ARRAY_SIZE 10000000

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    double *a = malloc(ARRAY_SIZE * sizeof(double));
    double *b = malloc(ARRAY_SIZE * sizeof(double));
    double *c = malloc(ARRAY_SIZE * sizeof(double));
    
    // Initialize
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }
    
    printf("Running memory bandwidth tests with %d elements...\n", ARRAY_SIZE);
    
    // Copy test
    double start = get_time();
    for (int i = 0; i < ARRAY_SIZE; i++) {
        c[i] = a[i];
    }
    double copy_time = get_time() - start;
    double copy_bandwidth = (2.0 * ARRAY_SIZE * sizeof(double)) / copy_time / 1e9;
    
    // Scale test
    start = get_time();
    for (int i = 0; i < ARRAY_SIZE; i++) {
        b[i] = 2.5 * c[i];
    }
    double scale_time = get_time() - start;
    double scale_bandwidth = (2.0 * ARRAY_SIZE * sizeof(double)) / scale_time / 1e9;
    
    // Add test
    start = get_time();
    for (int i = 0; i < ARRAY_SIZE; i++) {
        c[i] = a[i] + b[i];
    }
    double add_time = get_time() - start;
    double add_bandwidth = (3.0 * ARRAY_SIZE * sizeof(double)) / add_time / 1e9;
    
    printf("Copy: %.2f GB/s (%.6f seconds)\n", copy_bandwidth, copy_time);
    printf("Scale: %.2f GB/s (%.6f seconds)\n", scale_bandwidth, scale_time);
    printf("Add: %.2f GB/s (%.6f seconds)\n", add_bandwidth, add_time);
    
    double avg_bandwidth = (copy_bandwidth + scale_bandwidth + add_bandwidth) / 3.0;
    printf("Average bandwidth: %.2f GB/s\n", avg_bandwidth);
    
    // Save results
    FILE *f = fopen("memory_baseline_result.json", "w");
    fprintf(f, "{\n");
    fprintf(f, "  \"test_name\": \"memory_baseline_stream\",\n");
    fprintf(f, "  \"array_size\": %d,\n", ARRAY_SIZE);
    fprintf(f, "  \"copy_gbps\": %.2f,\n", copy_bandwidth);
    fprintf(f, "  \"scale_gbps\": %.2f,\n", scale_bandwidth);
    fprintf(f, "  \"add_gbps\": %.2f,\n", add_bandwidth);
    fprintf(f, "  \"average_gbps\": %.2f,\n", avg_bandwidth);
    fprintf(f, "  \"verification_value\": %.6f\n", c[ARRAY_SIZE-1]);
    fprintf(f, "}\n");
    fclose(f);
    
    free(a);
    free(b);
    free(c);
    
    return 0;
}
