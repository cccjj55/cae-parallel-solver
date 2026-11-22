/*
 * 2D P1 Finite Element Poisson Solver (-laplacian(u) = f on [0,1]^2, u=0 on boundary)
 *
 * Compares three execution modes over the exact same unstructured CSR mesh graph:
 *   serial       single-threaded element assembly + CG solver
 *   omp_atomic   OpenMP element assembly using #pragma omp atomic + OpenMP CG
 *   omp_colored  8-color lock-free OpenMP element assembly + OpenMP CG
 *
 * Verifies discrete L2 error against u_exact(x,y) = sin(pi*x)*sin(pi*y) and
 * reports stage timing breakdown (CSR assembly vs SpMV vs vector reductions).
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int n_dof;
    int nnz;
    int* row_ptr;
    int* col_idx;
    double* val;
} CSRMatrix;

typedef struct {
    int n_elem;
    int (*conn)[3];
    int color_ptr[9];
    int* color_elems;
} Mesh;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline double exact_u(double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y);
}

static inline double rhs_f(double x, double y) {
    return 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

static int interior_id(int ix, int iy, int nx) {
    if (ix <= 0 || ix >= nx || iy <= 0 || iy >= nx) return -1;
    return (iy - 1) * (nx - 1) + (ix - 1);
}

static void build_mesh(int nx, Mesh* mesh, double** node_x, double** node_y) {
    const int stride = nx + 1;
    const int n_nodes = stride * stride;
    *node_x = (double*)malloc((size_t)n_nodes * sizeof(double));
    *node_y = (double*)malloc((size_t)n_nodes * sizeof(double));
    const double h = 1.0 / (double)nx;
    for (int iy = 0; iy <= nx; ++iy) {
        for (int ix = 0; ix <= nx; ++ix) {
            const int id = iy * stride + ix;
            (*node_x)[id] = (double)ix * h;
            (*node_y)[id] = (double)iy * h;
        }
    }

    const int n_elem = 2 * nx * nx;
    mesh->n_elem = n_elem;
    mesh->conn = (int (*)[3])malloc((size_t)n_elem * sizeof(int[3]));
    int* elem_color = (int*)malloc((size_t)n_elem * sizeof(int));
    int color_counts[8] = {0};

    int e = 0;
    for (int iy = 0; iy < nx; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const int n00 = iy * stride + ix;
            const int n10 = n00 + 1;
            const int n01 = n00 + stride;
            const int n11 = n01 + 1;
            const int qcolor = (ix & 1) + 2 * (iy & 1);

            mesh->conn[e][0] = n00;
            mesh->conn[e][1] = n10;
            mesh->conn[e][2] = n11;
            elem_color[e] = 2 * qcolor;
            color_counts[elem_color[e]]++;
            e++;

            mesh->conn[e][0] = n00;
            mesh->conn[e][1] = n11;
            mesh->conn[e][2] = n01;
            elem_color[e] = 2 * qcolor + 1;
            color_counts[elem_color[e]]++;
            e++;
        }
    }

    mesh->color_ptr[0] = 0;
    for (int c = 0; c < 8; ++c) {
        mesh->color_ptr[c + 1] = mesh->color_ptr[c] + color_counts[c];
    }
    mesh->color_elems = (int*)malloc((size_t)n_elem * sizeof(int));
    int cursor[8];
    memcpy(cursor, mesh->color_ptr, 8 * sizeof(int));
    for (int i = 0; i < n_elem; ++i) {
        const int c = elem_color[i];
        mesh->color_elems[cursor[c]++] = i;
    }
    free(elem_color);
}

static void build_csr_pattern(int nx, CSRMatrix* mat) {
    const int m = nx - 1;
    const int n_dof = m * m;
    mat->n_dof = n_dof;
    mat->row_ptr = (int*)malloc((size_t)(n_dof + 1) * sizeof(int));

    // P1 triangular split along SW->NE gives at most 7 neighbors per interior node.
    const int dx[7] = {-1, 0, -1, 0, 1, 0, 1};
    const int dy[7] = {-1, -1, 0, 0, 0, 1, 1};

    int nnz = 0;
    mat->row_ptr[0] = 0;
    for (int iy = 1; iy < nx; ++iy) {
        for (int ix = 1; ix < nx; ++ix) {
            for (int k = 0; k < 7; ++k) {
                if (interior_id(ix + dx[k], iy + dy[k], nx) >= 0) nnz++;
            }
            const int row = interior_id(ix, iy, nx);
            mat->row_ptr[row + 1] = nnz;
        }
    }
    mat->nnz = nnz;
    mat->col_idx = (int*)malloc((size_t)nnz * sizeof(int));
    mat->val = (double*)calloc((size_t)nnz, sizeof(double));

    int pos = 0;
    for (int iy = 1; iy < nx; ++iy) {
        for (int ix = 1; ix < nx; ++ix) {
            for (int k = 0; k < 7; ++k) {
                const int col = interior_id(ix + dx[k], iy + dy[k], nx);
                if (col >= 0) mat->col_idx[pos++] = col;
            }
        }
    }
}

static inline int find_csr_slot(const CSRMatrix* mat, int row, int col) {
    for (int k = mat->row_ptr[row]; k < mat->row_ptr[row + 1]; ++k) {
        if (mat->col_idx[k] == col) return k;
    }
    return -1;
}

static inline void elem_stiffness(
    const int tri[3], const double* nx_coord, const double* ny_coord,
    double ke[3][3], double fe[3]) {
    const double x0 = nx_coord[tri[0]], y0 = ny_coord[tri[0]];
    const double x1 = nx_coord[tri[1]], y1 = ny_coord[tri[1]];
    const double x2 = nx_coord[tri[2]], y2 = ny_coord[tri[2]];

    const double twice_area = fabs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    const double inv_4a = 1.0 / (2.0 * twice_area);

    const double b[3] = {y1 - y2, y2 - y0, y0 - y1};
    const double c[3] = {x2 - x1, x0 - x2, x1 - x0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ke[i][j] = (b[i] * b[j] + c[i] * c[j]) * inv_4a;
        }
    }

    const double f01 = rhs_f(0.5 * (x0 + x1), 0.5 * (y0 + y1));
    const double f12 = rhs_f(0.5 * (x1 + x2), 0.5 * (y1 + y2));
    const double f20 = rhs_f(0.5 * (x2 + x0), 0.5 * (y2 + y0));
    const double w = (0.5 * twice_area) / 6.0;
    fe[0] = w * (f01 + f20);
    fe[1] = w * (f01 + f12);
    fe[2] = w * (f12 + f20);
}

static inline int global_to_dof(int node, int nx) {
    const int stride = nx + 1;
    return interior_id(node % stride, node / stride, nx);
}

static void assemble_serial(
    int nx, const Mesh* mesh, const double* node_x, const double* node_y,
    CSRMatrix* mat, double* rhs) {
    memset(mat->val, 0, (size_t)mat->nnz * sizeof(double));
    memset(rhs, 0, (size_t)mat->n_dof * sizeof(double));

    for (int e = 0; e < mesh->n_elem; ++e) {
        double ke[3][3], fe[3];
        elem_stiffness(mesh->conn[e], node_x, node_y, ke, fe);
        int dofs[3];
        for (int a = 0; a < 3; ++a) dofs[a] = global_to_dof(mesh->conn[e][a], nx);

        for (int a = 0; a < 3; ++a) {
            if (dofs[a] < 0) continue;
            rhs[dofs[a]] += fe[a];
            for (int b = 0; b < 3; ++b) {
                if (dofs[b] < 0) continue;
                const int slot = find_csr_slot(mat, dofs[a], dofs[b]);
                mat->val[slot] += ke[a][b];
            }
        }
    }
}

static void assemble_omp_atomic(
    int nx, const Mesh* mesh, const double* node_x, const double* node_y,
    CSRMatrix* mat, double* rhs) {
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < mat->nnz; ++k) mat->val[k] = 0.0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < mat->n_dof; ++i) rhs[i] = 0.0;

    #pragma omp parallel for schedule(static)
    for (int e = 0; e < mesh->n_elem; ++e) {
        double ke[3][3], fe[3];
        elem_stiffness(mesh->conn[e], node_x, node_y, ke, fe);
        int dofs[3];
        for (int a = 0; a < 3; ++a) dofs[a] = global_to_dof(mesh->conn[e][a], nx);

        for (int a = 0; a < 3; ++a) {
            if (dofs[a] < 0) continue;
            #pragma omp atomic
            rhs[dofs[a]] += fe[a];
            for (int b = 0; b < 3; ++b) {
                if (dofs[b] < 0) continue;
                const int slot = find_csr_slot(mat, dofs[a], dofs[b]);
                #pragma omp atomic
                mat->val[slot] += ke[a][b];
            }
        }
    }
}

static void assemble_omp_colored(
    int nx, const Mesh* mesh, const double* node_x, const double* node_y,
    CSRMatrix* mat, double* rhs) {
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < mat->nnz; ++k) mat->val[k] = 0.0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < mat->n_dof; ++i) rhs[i] = 0.0;

    for (int c = 0; c < 8; ++c) {
        const int begin = mesh->color_ptr[c];
        const int end = mesh->color_ptr[c + 1];
        #pragma omp parallel for schedule(static)
        for (int idx = begin; idx < end; ++idx) {
            const int e = mesh->color_elems[idx];
            double ke[3][3], fe[3];
            elem_stiffness(mesh->conn[e], node_x, node_y, ke, fe);
            int dofs[3];
            for (int a = 0; a < 3; ++a) dofs[a] = global_to_dof(mesh->conn[e][a], nx);

            for (int a = 0; a < 3; ++a) {
                if (dofs[a] < 0) continue;
                rhs[dofs[a]] += fe[a];
                for (int b = 0; b < 3; ++b) {
                    if (dofs[b] < 0) continue;
                    const int slot = find_csr_slot(mat, dofs[a], dofs[b]);
                    mat->val[slot] += ke[a][b];
                }
            }
        }
    }
}

static int cg_solve(
    const CSRMatrix* mat, const double* rhs, double* u,
    int use_omp, int max_iter, double tol,
    double* spmv_time_out, double* vec_time_out) {
    const int n = mat->n_dof;
    double* r = (double*)malloc((size_t)n * sizeof(double));
    double* p = (double*)malloc((size_t)n * sizeof(double));
    double* Ap = (double*)malloc((size_t)n * sizeof(double));

    double rzold = 0.0;
    if (use_omp) {
        #pragma omp parallel for schedule(static) reduction(+:rzold)
        for (int i = 0; i < n; ++i) {
            u[i] = 0.0;
            r[i] = rhs[i];
            p[i] = rhs[i];
            rzold += r[i] * r[i];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            u[i] = 0.0;
            r[i] = rhs[i];
            p[i] = rhs[i];
            rzold += r[i] * r[i];
        }
    }

    const double tol2 = tol * tol * rzold;
    double spmv_t = 0.0, vec_t = 0.0;
    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        if (rzold <= tol2) break;

        const double t0 = now_sec();
        if (use_omp) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int k = mat->row_ptr[i]; k < mat->row_ptr[i + 1]; ++k) {
                    sum += mat->val[k] * p[mat->col_idx[k]];
                }
                Ap[i] = sum;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int k = mat->row_ptr[i]; k < mat->row_ptr[i + 1]; ++k) {
                    sum += mat->val[k] * p[mat->col_idx[k]];
                }
                Ap[i] = sum;
            }
        }
        const double t1 = now_sec();
        spmv_t += (t1 - t0);

        double pAp = 0.0;
        if (use_omp) {
            #pragma omp parallel for schedule(static) reduction(+:pAp)
            for (int i = 0; i < n; ++i) pAp += p[i] * Ap[i];
        } else {
            for (int i = 0; i < n; ++i) pAp += p[i] * Ap[i];
        }
        const double alpha = rzold / pAp;

        double rznew = 0.0;
        if (use_omp) {
            #pragma omp parallel for schedule(static) reduction(+:rznew)
            for (int i = 0; i < n; ++i) {
                u[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
                rznew += r[i] * r[i];
            }
        } else {
            for (int i = 0; i < n; ++i) {
                u[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
                rznew += r[i] * r[i];
            }
        }
        const double beta = rznew / rzold;
        if (use_omp) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        } else {
            for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        }
        rzold = rznew;
        vec_t += (now_sec() - t1);
    }

    free(r);
    free(p);
    free(Ap);
    *spmv_time_out = spmv_t;
    *vec_time_out = vec_t;
    return iter;
}

static double compute_l2_error(int nx, const double* u) {
    const double h = 1.0 / (double)nx;
    double err2 = 0.0;
    for (int iy = 1; iy < nx; ++iy) {
        const double y = (double)iy * h;
        for (int ix = 1; ix < nx; ++ix) {
            const double x = (double)ix * h;
            const int row = interior_id(ix, iy, nx);
            const double diff = u[row] - exact_u(x, y);
            err2 += diff * diff * h * h;
        }
    }
    return sqrt(err2);
}

typedef struct {
    double total_s;
    double asm_s;
    double spmv_s;
    double vec_s;
    double l2_err;
    int iters;
} RunStats;

static RunStats run_variant(
    const char* mode, int threads, int nx, const Mesh* mesh,
    const double* node_x, const double* node_y, CSRMatrix* mat,
    double* rhs, double* u, int reps) {
    const int use_omp = strcmp(mode, "serial") != 0;
    if (use_omp) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }

    RunStats best = {.total_s = 1e30};
    for (int r = 0; r <= reps; ++r) {
        const double t0 = now_sec();
        if (strcmp(mode, "serial") == 0) {
            assemble_serial(nx, mesh, node_x, node_y, mat, rhs);
        } else if (strcmp(mode, "omp_atomic") == 0) {
            assemble_omp_atomic(nx, mesh, node_x, node_y, mat, rhs);
        } else {
            assemble_omp_colored(nx, mesh, node_x, node_y, mat, rhs);
        }
        const double t1 = now_sec();

        double spmv_s = 0.0, vec_s = 0.0;
        const int iters = cg_solve(mat, rhs, u, use_omp, nx * 8, 1e-10, &spmv_s, &vec_s);
        const double t2 = now_sec();
        if (r == 0) continue;  // untimed warmup

        const double total = t2 - t0;
        if (total < best.total_s) {
            best.total_s = total;
            best.asm_s = t1 - t0;
            best.spmv_s = spmv_s;
            best.vec_s = vec_s;
            best.iters = iters;
            best.l2_err = compute_l2_error(nx, u);
        }
    }
    return best;
}

static int run_tests(void) {
    // 1. Verify O(h^2) convergence: doubling nx shrinks L2 error by ~4x.
    double errs[2] = {0.0, 0.0};
    const int sizes[2] = {32, 64};
    for (int s = 0; s < 2; ++s) {
        const int nx = sizes[s];
        Mesh mesh;
        double *node_x, *node_y;
        CSRMatrix mat;
        build_mesh(nx, &mesh, &node_x, &node_y);
        build_csr_pattern(nx, &mat);
        double* rhs = (double*)malloc((size_t)mat.n_dof * sizeof(double));
        double* u_ser = (double*)malloc((size_t)mat.n_dof * sizeof(double));
        double* u_atm = (double*)malloc((size_t)mat.n_dof * sizeof(double));
        double* u_col = (double*)malloc((size_t)mat.n_dof * sizeof(double));

        RunStats rs = run_variant("serial", 1, nx, &mesh, node_x, node_y, &mat, rhs, u_ser, 1);
        RunStats ra = run_variant("omp_atomic", 4, nx, &mesh, node_x, node_y, &mat, rhs, u_atm, 1);
        RunStats rc = run_variant("omp_colored", 4, nx, &mesh, node_x, node_y, &mat, rhs, u_col, 1);
        errs[s] = rs.l2_err;

        double max_diff_a = 0.0, max_diff_c = 0.0;
        for (int i = 0; i < mat.n_dof; ++i) {
            double da = fabs(u_atm[i] - u_ser[i]);
            double dc = fabs(u_col[i] - u_ser[i]);
            if (da > max_diff_a) max_diff_a = da;
            if (dc > max_diff_c) max_diff_c = dc;
        }
        printf("nx=%d dof=%d nnz=%d L2=%.3e iters=%d diff_atomic=%.2e diff_colored=%.2e\n",
               nx, mat.n_dof, mat.nnz, rs.l2_err, rs.iters, max_diff_a, max_diff_c);
        if (max_diff_a > 1e-8 || max_diff_c > 1e-8 || fabs(ra.l2_err - rs.l2_err) > 1e-8 ||
            fabs(rc.l2_err - rs.l2_err) > 1e-8) {
            fprintf(stderr, "FAIL: parallel solution diverged from serial\n");
            return 1;
        }
        free(rhs);
        free(u_ser);
        free(u_atm);
        free(u_col);
        free(mat.row_ptr);
        free(mat.col_idx);
        free(mat.val);
        free(mesh.conn);
        free(mesh.color_elems);
        free(node_x);
        free(node_y);
    }
    const double ratio = errs[0] / errs[1];
    printf("O(h^2) convergence ratio (nx=32 -> 64): %.2fx\n", ratio);
    if (ratio < 3.7 || ratio > 4.3) {
        fprintf(stderr, "FAIL: expected ~4.0x L2 error reduction, got %.2fx\n", ratio);
        return 1;
    }
    printf("ALL TESTS PASSED\n");
    return 0;
}

int main(int argc, char** argv) {
    int nx = 400;
    int reps = 3;
    int csv = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--test") == 0) return run_tests();
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        if (strcmp(argv[i], "--csv") == 0) csv = 1;
    }

    Mesh mesh;
    double *node_x, *node_y;
    CSRMatrix mat;
    build_mesh(nx, &mesh, &node_x, &node_y);
    build_csr_pattern(nx, &mat);
    double* rhs = (double*)malloc((size_t)mat.n_dof * sizeof(double));
    double* u_ref = (double*)malloc((size_t)mat.n_dof * sizeof(double));
    double* u = (double*)malloc((size_t)mat.n_dof * sizeof(double));

    const int max_threads = omp_get_max_threads();
    const double bytes_per_spmv = 12.0 * (double)mat.nnz + 16.0 * (double)mat.n_dof;

    if (csv) {
        printf("mode,threads,nx,dof,nnz,iters,total_s,asm_s,spmv_s,vec_s,spmv_gbps,l2_err\n");
    }

    RunStats base = run_variant("serial", 1, nx, &mesh, node_x, node_y, &mat, rhs, u_ref, reps);
    if (csv) {
        const double gbps = (bytes_per_spmv * (double)base.iters / base.spmv_s) / 1e9;
        printf("serial,1,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.4e\n",
               nx, mat.n_dof, mat.nnz, base.iters, base.total_s, base.asm_s,
               base.spmv_s, base.vec_s, gbps, base.l2_err);
    }

    const int thread_counts[] = {1, 2, 4, 6, 12};
    for (int ti = 0; ti < 5; ++ti) {
        const int th = thread_counts[ti];
        if (th > max_threads) continue;
        const char* modes[] = {"omp_atomic", "omp_colored"};
        for (int mi = 0; mi < 2; ++mi) {
            if (th == 1 && mi == 0) continue;
            RunStats st = run_variant(modes[mi], th, nx, &mesh, node_x, node_y, &mat, rhs, u, reps);
            double max_diff = 0.0;
            for (int i = 0; i < mat.n_dof; ++i) {
                double d = fabs(u[i] - u_ref[i]);
                if (d > max_diff) max_diff = d;
            }
            if (max_diff > 1e-8) {
                fprintf(stderr, "FAIL: %s t=%d diverged (max_diff=%.2e)\n", modes[mi], th, max_diff);
                return 2;
            }
            const double gbps = (bytes_per_spmv * (double)st.iters / st.spmv_s) / 1e9;
            if (csv) {
                printf("%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.4e\n",
                       modes[mi], th, nx, mat.n_dof, mat.nnz, st.iters,
                       st.total_s, st.asm_s, st.spmv_s, st.vec_s, gbps, st.l2_err);
            } else {
                printf("%-11s t=%2d  total=%7.2f ms (x%.2f)  asm=%6.2f ms  spmv=%6.2f ms (%.2f GB/s)  vec=%6.2f ms\n",
                       modes[mi], th, st.total_s * 1e3, base.total_s / st.total_s,
                       st.asm_s * 1e3, st.spmv_s * 1e3, gbps, st.vec_s * 1e3);
            }
        }
    }

    free(rhs);
    free(u_ref);
    free(u);
    free(mat.row_ptr);
    free(mat.col_idx);
    free(mat.val);
    free(mesh.conn);
    free(mesh.color_elems);
    free(node_x);
    free(node_y);
    return 0;
}
