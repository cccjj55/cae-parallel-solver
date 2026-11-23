/* End-to-end conjugate gradient on the GPU, on the CSR matrix the CPU assembles.
 *
 * Plain CG (no preconditioner) with the same tolerance the CPU solver uses, so the iteration
 * counts are comparable. Two timings are reported separately:
 *
 *   spmv_only  the same number of SpMVs as the CG run performs, timed alone
 *   cg_total   the whole solve
 *
 * The difference is the vector work (3 dots + 3 axpy-class kernels per iteration), which is where
 * a GPU loses its SpMV advantage: a dot product is a reduction, and a reduction is latency, not
 * bandwidth.
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
    int n, nnz;
    int *row_ptr, *col_idx;
    double *val, *b;
} HostCsr;

static HostCsr load_csr(const char* path) {
    HostCsr h = {0, 0, NULL, NULL, NULL, NULL};
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CSR1", 4) != 0) { fprintf(stderr, "not CSR1\n"); exit(2); }
    if (fread(&h.n, 4, 1, f) != 1 || fread(&h.nnz, 4, 1, f) != 1) exit(2);
    h.row_ptr = (int*)malloc((size_t)(h.n + 1) * sizeof(int));
    h.col_idx = (int*)malloc((size_t)h.nnz * sizeof(int));
    h.val = (double*)malloc((size_t)h.nnz * sizeof(double));
    h.b = (double*)malloc((size_t)h.n * sizeof(double));
    if (fread(h.row_ptr, 4, (size_t)h.n + 1, f) != (size_t)h.n + 1) exit(2);
    if (fread(h.col_idx, 4, (size_t)h.nnz, f) != (size_t)h.nnz) exit(2);
    if (fread(h.val, 8, (size_t)h.nnz, f) != (size_t)h.nnz) exit(2);
    if (fread(h.b, 8, (size_t)h.n, f) != (size_t)h.n) exit(2);
    fclose(f);
    return h;
}

/* ---- kernels ---------------------------------------------------------- */

__global__ void k_spmv(int n, const int* row, const int* col, const double* val,
                       const double* x, double* y) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    double s = 0.0;
    for (int k = row[r]; k < row[r + 1]; ++k) s += val[k] * x[col[k]];
    y[r] = s;
}

__global__ void k_dot_partial(const double* a, const double* b, double* out, int n) {
    __shared__ double s[256];
    double acc = 0.0;
    const int tid = threadIdx.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) acc += a[i] * b[i];
    s[tid] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

__global__ void k_axpy(double* y, const double* x, double a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void k_axpby(double* y, const double* x, double a, double b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * y[i] + b * x[i];
}

__global__ void k_scale_copy(double* dst, const double* src, double a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a * src[i];
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "build/m400.csr";
    const int max_iters = argc > 2 ? atoi(argv[2]) : 5000;
    const double tol = argc > 3 ? atof(argv[3]) : 1e-10;

    HostCsr h = load_csr(path);
    const int n = h.n;

    int *d_row, *d_col;
    double *d_val, *d_b, *d_x, *d_r, *d_p, *d_ap, *d_part;
    CHECK(cudaMalloc(&d_row, (size_t)(n + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_col, (size_t)h.nnz * sizeof(int)));
    CHECK(cudaMalloc(&d_val, (size_t)h.nnz * sizeof(double)));
    CHECK(cudaMalloc(&d_b, (size_t)n * sizeof(double)));
    CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
    CHECK(cudaMalloc(&d_r, (size_t)n * sizeof(double)));
    CHECK(cudaMalloc(&d_p, (size_t)n * sizeof(double)));
    CHECK(cudaMalloc(&d_ap, (size_t)n * sizeof(double)));
    const int blocks = 256;
    CHECK(cudaMalloc(&d_part, (size_t)blocks * sizeof(double)));

    CHECK(cudaMemcpy(d_row, h.row_ptr, (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col, h.col_idx, (size_t)h.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_val, h.val, (size_t)h.nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h.b, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (n + block - 1) / block;

    /* norm of b, needed for the relative stopping test */
    double bnorm = 0.0;
    for (int i = 0; i < n; ++i) bnorm += h.b[i] * h.b[i];
    bnorm = sqrt(bnorm);

    /* ---- Phase 1: pure SpMV loop, timed alone ------------------------- */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    /* ---- Phase 2: CG -------------------------------------------------- */
    CHECK(cudaMemset(d_x, 0, (size_t)n * sizeof(double)));
    CHECK(cudaMemcpy(d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(d_p, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));

    double* host_part = (double*)malloc((size_t)blocks * sizeof(double));

    cudaEventRecord(t0);
    int iters = 0;
    for (int it = 0; it < max_iters; ++it) {
        k_spmv<<<grid, block>>>(n, d_row, d_col, d_val, d_p, d_ap);
        k_dot_partial<<<blocks, block>>>(d_p, d_ap, d_part, n);
        CHECK(cudaMemcpy(host_part, d_part, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost));
        double pAp = 0.0;
        for (int i = 0; i < blocks; ++i) pAp += host_part[i];

        k_dot_partial<<<blocks, block>>>(d_r, d_r, d_part, n);
        CHECK(cudaMemcpy(host_part, d_part, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost));
        double rs = 0.0;
        for (int i = 0; i < blocks; ++i) rs += host_part[i];

        const double alpha = rs / pAp;
        k_axpy<<<grid, block>>>(d_x, d_p, alpha, n);
        k_axpy<<<grid, block>>>(d_r, d_ap, -alpha, n);

        k_dot_partial<<<blocks, block>>>(d_r, d_r, d_part, n);
        CHECK(cudaMemcpy(host_part, d_part, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost));
        double rsnew = 0.0;
        for (int i = 0; i < blocks; ++i) rsnew += host_part[i];

        iters = it + 1;
        if (sqrt(rsnew) < tol * bnorm) break;
        const double beta = rsnew / rs;
        k_axpby<<<grid, block>>>(d_p, d_r, beta, 1.0, n);
    }
    cudaEventRecord(t1);
    CHECK(cudaEventSynchronize(t1));
    float cg_ms = 0.0f;
    cudaEventElapsedTime(&cg_ms, t0, t1);

    /* pure SpMV loop over the same iteration count, for the phase split.
     * p is refreshed from b first: after convergence the vectors hold denormals, and denormal
     * arithmetic on the GPU is slow enough to distort the measurement. */
    CHECK(cudaMemcpy(d_p, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
    cudaEventRecord(t0);
    for (int it = 0; it < iters; ++it) k_spmv<<<grid, block>>>(n, d_row, d_col, d_val, d_p, d_ap);
    cudaEventRecord(t1);
    CHECK(cudaEventSynchronize(t1));
    float spmv_ms = 0.0f;
    cudaEventElapsedTime(&spmv_ms, t0, t1);

    /* residual check on the host */
    double* x = (double*)malloc((size_t)n * sizeof(double));
    CHECK(cudaMemcpy(x, d_x, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    double rnorm2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = h.row_ptr[i]; k < h.row_ptr[i + 1]; ++k) s += h.val[k] * x[h.col_idx[k]];
        const double d = h.b[i] - s;
        rnorm2 += d * d;
    }
    const double rel_res = sqrt(rnorm2) / bnorm;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += x[i];

    printf("file,n,nnz,iters,cg_ms,spmv_ms,vec_ms,cg_ms_per_iter,rel_residual,sum_x\n");
    printf("%s,%d,%d,%d,%.3f,%.3f,%.3f,%.4f,%.3e,%.9f\n", path, n, h.nnz, iters, cg_ms, spmv_ms,
           cg_ms - spmv_ms, cg_ms / (double)iters, rel_res, sum);

    free(x); free(host_part);
    free(h.row_ptr); free(h.col_idx); free(h.val); free(h.b);
    return 0;
}
