/* Conjugate gradient on the GPU, with the loop kept on the device.
 *
 * v1 of this file synchronised with the host three times per iteration (three dot products, each a
 * device->host copy). At 159k DOF that latency dominated: 0.44x of a 12-thread CPU.
 *
 * This version removes the round trips:
 *   - the two dots that are actually needed per iteration (p.Ap, r.r) are reduced on the device
 *   - alpha and beta are computed by a one-thread kernel, so nothing comes back to the host
 *   - the axpy/axpby kernels read the scalar from device memory instead of a kernel argument
 *   - the residual is only inspected every --sync-every iterations, which is the only host sync left
 *
 * The maths is unchanged; the schedule is what changed.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
    } while (0)

typedef struct { int n, nnz; int *row_ptr, *col_idx; double *val, *b; } HostCsr;

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

/* finish a partial-sum array down to one scalar, on the device */
__global__ void k_reduce_finish(double* v, int count) {
    __shared__ double s[256];
    double acc = 0.0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) acc += v[i];
    __syncthreads();
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) v[0] = s[0];
}

/* y += a*x, with a living on the device: no host round trip */
__global__ void k_axpy_dev(double* y, const double* x, const double* a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += (*a) * x[i];
}

/* y -= a*x, a on the device */
__global__ void k_axpy_neg_dev(double* y, const double* x, const double* a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] -= (*a) * x[i];
}

/* y = beta*y + x, beta on the device */
__global__ void k_axpby_dev(double* y, const double* x, const double* beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (*beta) * y[i] + x[i];
}

/* alpha = rsold / pAp */
__global__ void k_alpha(const double* rsold, const double* pAp, double* alpha) {
    *alpha = (*rsold) / (*pAp);
}

/* beta = rsnew / rsold; rsold <- rsnew */
__global__ void k_beta(const double* rsnew, double* rsold, double* beta) {
    *beta = (*rsnew) / (*rsold);
    *rsold = *rsnew;
}

__global__ void k_zero(double* v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = 0.0;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "build/m400.csr";
    const int max_iters = argc > 2 ? atoi(argv[2]) : 5000;
    const double tol = argc > 3 ? atof(argv[3]) : 1e-10;
    int sync_every = argc > 4 ? atoi(argv[4]) : 20;

    HostCsr h = load_csr(path);
    const int n = h.n;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const int blocks = 256;

    int *d_row, *d_col;
    double *d_val, *d_b, *d_x, *d_r, *d_p, *d_ap, *d_part, *d_scal;
    CHECK(cudaMalloc(&d_row, (size_t)(n + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_col, (size_t)h.nnz * sizeof(int)));
    CHECK(cudaMalloc(&d_val, (size_t)h.nnz * sizeof(double)));
    CHECK(cudaMalloc(&d_b, (size_t)n * sizeof(double)));
    CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
    CHECK(cudaMalloc(&d_r, (size_t)n * sizeof(double)));
    CHECK(cudaMalloc(&d_p, (size_t)n * sizeof(double)));
    CHECK(cudaMalloc(&d_ap, (size_t)n * sizeof(double)));
    CHECK(cudaMalloc(&d_part, (size_t)blocks * sizeof(double)));
    CHECK(cudaMalloc(&d_scal, 5 * sizeof(double))); /* alpha, beta, rsold, rsnew, pAp */

    CHECK(cudaMemcpy(d_row, h.row_ptr, (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col, h.col_idx, (size_t)h.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_val, h.val, (size_t)h.nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h.b, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));

    double* alpha = d_scal + 0;
    double* beta = d_scal + 1;
    double* rsold = d_scal + 2;
    double* rsnew = d_scal + 3;
    double* pap = d_scal + 4;

    double bnorm = 0.0;
    for (int i = 0; i < n; ++i) bnorm += h.b[i] * h.b[i];
    bnorm = sqrt(bnorm);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    /* x = 0, r = b, p = b, rsold = b.b */
    k_zero<<<grid, block>>>(d_x, n);
    CHECK(cudaMemcpy(d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(d_p, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
    k_dot_partial<<<blocks, block>>>(d_b, d_b, d_part, n);
    k_reduce_finish<<<1, 256>>>(d_part, blocks);
    CHECK(cudaMemcpy(rsold, d_part, sizeof(double), cudaMemcpyDeviceToDevice));

    cudaEventRecord(t0);
    int iters = 0;
    double residual = 1.0;
    for (int it = 0; it < max_iters; ++it) {
        k_spmv<<<grid, block>>>(n, d_row, d_col, d_val, d_p, d_ap);

        /* pAp = p.Ap, reduced on the device */
        k_dot_partial<<<blocks, block>>>(d_p, d_ap, d_part, n);
        k_reduce_finish<<<1, 256>>>(d_part, blocks);
        CHECK(cudaMemcpy(pap, d_part, sizeof(double), cudaMemcpyDeviceToDevice));

        /* alpha = rsold / pAp, on the device: no host round trip */
        k_alpha<<<1, 1>>>(rsold, pap, alpha);

        k_axpy_dev<<<grid, block>>>(d_x, d_p, alpha, n);
        k_axpy_neg_dev<<<grid, block>>>(d_r, d_ap, alpha, n);
        iters = it + 1;

        /* r.r */
        k_dot_partial<<<blocks, block>>>(d_r, d_r, d_part, n);
        k_reduce_finish<<<1, 256>>>(d_part, blocks);
        CHECK(cudaMemcpy(rsnew, d_part, sizeof(double), cudaMemcpyDeviceToDevice));

        if ((it + 1) % sync_every == 0 || it + 1 == max_iters) {
            double rn = 0.0;
            CHECK(cudaMemcpy(&rn, rsnew, sizeof(double), cudaMemcpyDeviceToHost));
            residual = sqrt(rn) / bnorm;
            if (residual < tol) break;
        }

        k_beta<<<1, 1>>>(rsnew, rsold, beta);
        k_axpby_dev<<<grid, block>>>(d_p, d_r, beta, n);
    }
    cudaEventRecord(t1);
    CHECK(cudaEventSynchronize(t1));
    float cg_ms = 0.0f;
    cudaEventElapsedTime(&cg_ms, t0, t1);

    double* x = (double*)malloc((size_t)n * sizeof(double));
    CHECK(cudaMemcpy(x, d_x, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    double rnorm2 = 0.0, sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = h.row_ptr[i]; k < h.row_ptr[i + 1]; ++k) s += h.val[k] * x[h.col_idx[k]];
        const double d = h.b[i] - s;
        rnorm2 += d * d;
        sum += x[i];
    }

    printf("file,n,nnz,iters,cg_ms,ms_per_iter,rel_residual,sum_x,sync_every\n");
    printf("%s,%d,%d,%d,%.3f,%.4f,%.3e,%.9f,%d\n", path, n, h.nnz, iters, cg_ms,
           cg_ms / (double)iters, sqrt(rnorm2) / bnorm, sum, sync_every);

    free(x);
    free(h.row_ptr); free(h.col_idx); free(h.val); free(h.b);
    return 0;
}
