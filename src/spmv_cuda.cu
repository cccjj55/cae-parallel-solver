/* CSR SpMV on the GPU, against the same matrix the CPU solver assembles.
 *
 * Three kernels, because "GPU is faster" is not a measurement:
 *   scalar      one thread per row (fine for a 7-nnz row, wasteful for a 200-nnz row)
 *   two_rows    two rows per thread, for instruction-level parallelism
 *   warp        one warp per row (wins only when rows are long and skewed)
 *
 * Bandwidth accounting matches the CPU side exactly: 12 bytes per nnz (8 for the
 * value + 4 for the column index) plus 16 bytes per row (read x, write y).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

typedef struct {
    int n;
    int nnz;
    int* row_ptr;
    int* col_idx;
    double* val;
    double* x;
} HostCsr;

static HostCsr load_csr(const char* path) {
    HostCsr h = {0, 0, NULL, NULL, NULL, NULL};
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CSR1", 4) != 0) {
        fprintf(stderr, "%s: not a CSR1 file\n", path);
        exit(2);
    }
    if (fread(&h.n, sizeof(int), 1, f) != 1) exit(2);
    if (fread(&h.nnz, sizeof(int), 1, f) != 1) exit(2);
    h.row_ptr = (int*)malloc((size_t)(h.n + 1) * sizeof(int));
    h.col_idx = (int*)malloc((size_t)h.nnz * sizeof(int));
    h.val = (double*)malloc((size_t)h.nnz * sizeof(double));
    h.x = (double*)malloc((size_t)h.n * sizeof(double));
    if (fread(h.row_ptr, sizeof(int), (size_t)h.n + 1, f) != (size_t)h.n + 1) exit(2);
    if (fread(h.col_idx, sizeof(int), (size_t)h.nnz, f) != (size_t)h.nnz) exit(2);
    if (fread(h.val, sizeof(double), (size_t)h.nnz, f) != (size_t)h.nnz) exit(2);
    if (fread(h.x, sizeof(double), (size_t)h.n, f) != (size_t)h.n) exit(2);
    fclose(f);
    return h;
}

__global__ void spmv_scalar(int n, const int* row_ptr, const int* col, const double* val,
                            const double* x, double* y) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double sum = 0.0;
    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) sum += val[k] * x[col[k]];
    y[row] = sum;
}

__global__ void spmv_two_rows(int n, const int* row_ptr, const int* col, const double* val,
                              const double* x, double* y) {
    const int r0 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (r0 >= n) return;
    double s0 = 0.0;
    for (int k = row_ptr[r0]; k < row_ptr[r0 + 1]; ++k) s0 += val[k] * x[col[k]];
    y[r0] = s0;
    const int r1 = r0 + 1;
    if (r1 >= n) return;
    double s1 = 0.0;
    for (int k = row_ptr[r1]; k < row_ptr[r1 + 1]; ++k) s1 += val[k] * x[col[k]];
    y[r1] = s1;
}

__global__ void spmv_warp(int n, const int* row_ptr, const int* col, const double* val,
                          const double* x, double* y) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= n) return;
    double sum = 0.0;
    for (int k = row_ptr[warp] + lane; k < row_ptr[warp + 1]; k += 32) sum += val[k] * x[col[k]];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[warp] = sum;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "build/m400.csr";
    int iters = argc > 2 ? atoi(argv[2]) : 30;

    HostCsr h = load_csr(path);
    const double bytes = 12.0 * (double)h.nnz + 16.0 * (double)h.n;

    /* CPU reference */
    double* y_cpu = (double*)malloc((size_t)h.n * sizeof(double));
    for (int r = 0; r < h.n; ++r) {
        double s = 0.0;
        for (int k = h.row_ptr[r]; k < h.row_ptr[r + 1]; ++k) s += h.val[k] * h.x[h.col_idx[k]];
        y_cpu[r] = s;
    }

    int *d_row, *d_col;
    double *d_val, *d_x, *d_y;
    CHECK(cudaMalloc(&d_row, (size_t)(h.n + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_col, (size_t)h.nnz * sizeof(int)));
    CHECK(cudaMalloc(&d_val, (size_t)h.nnz * sizeof(double)));
    CHECK(cudaMalloc(&d_x, (size_t)h.n * sizeof(double)));
    CHECK(cudaMalloc(&d_y, (size_t)h.n * sizeof(double)));
    CHECK(cudaMemcpy(d_row, h.row_ptr, (size_t)(h.n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col, h.col_idx, (size_t)h.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_val, h.val, (size_t)h.nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, h.x, (size_t)h.n * sizeof(double), cudaMemcpyHostToDevice));

    double* y_gpu = (double*)malloc((size_t)h.n * sizeof(double));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("size,n,nnz,kernel,blocks,threads,ms,gbps,maxdiff\n");

    const char* names[3] = {"scalar", "two_rows", "warp"};
    for (int k = 0; k < 3; ++k) {
        int block = 256;
        int grid;
        if (k == 2) {
            grid = (h.n * 32 + block - 1) / block; /* one warp per row */
        } else if (k == 1) {
            grid = ((h.n + 1) / 2 + block - 1) / block;
        } else {
            grid = (h.n + block - 1) / block;
        }

        /* warmup */
        for (int w = 0; w < 3; ++w) {
            if (k == 0) spmv_scalar<<<grid, block>>>(h.n, d_row, d_col, d_val, d_x, d_y);
            else if (k == 1) spmv_two_rows<<<grid, block>>>(h.n, d_row, d_col, d_val, d_x, d_y);
            else spmv_warp<<<grid, block>>>(h.n, d_row, d_col, d_val, d_x, d_y);
        }
        CHECK(cudaDeviceSynchronize());

        cudaEventRecord(start);
        for (int it = 0; it < iters; ++it) {
            if (k == 0) spmv_scalar<<<grid, block>>>(h.n, d_row, d_col, d_val, d_x, d_y);
            else if (k == 1) spmv_two_rows<<<grid, block>>>(h.n, d_row, d_col, d_val, d_x, d_y);
            else spmv_warp<<<grid, block>>>(h.n, d_row, d_col, d_val, d_x, d_y);
        }
        cudaEventRecord(stop);
        CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        const double per_iter = (double)ms / iters;

        CHECK(cudaMemcpy(y_gpu, d_y, (size_t)h.n * sizeof(double), cudaMemcpyDeviceToHost));
        double maxdiff = 0.0;
        for (int r = 0; r < h.n; ++r) {
            const double d = fabs(y_gpu[r] - y_cpu[r]);
            if (d > maxdiff) maxdiff = d;
        }
        printf("%s,%d,%d,%s,%d,%d,%.4f,%.2f,%.3e\n", path, h.n, h.nnz, names[k], grid, block,
               per_iter, bytes / (per_iter * 1e-3) / 1e9, maxdiff);
    }

    cudaFree(d_row); cudaFree(d_col); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    free(y_cpu); free(y_gpu);
    free(h.row_ptr); free(h.col_idx); free(h.val); free(h.x);
    return 0;
}
