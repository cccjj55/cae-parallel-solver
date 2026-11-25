# cae-parallel-solver

2D P1 Finite Element Poisson solver (`-∇²u = f` on `[0,1]²`, `u = 0` on `∂Ω`) in C11 + OpenMP,
built to answer one CAE performance engineering question: **where does solver time go, and why do
more threads stop helping once the sparse matrix leaves L3 cache?**

![2D P1 FEM Solver: L3-Resident vs DRAM-Bound Breakdown](docs/fem_breakdown.svg)

## What is inside

- **Unstructured CSR assembly from triangular P1 elements**: computes exact `3×3` local stiffness
  matrices and 3-point edge-midpoint load vectors over `2·nx²` triangles and scatters into CSR.
- **Three execution modes over identical CSR sparsity**:
  - `serial` — single-threaded assembly + Conjugate Gradient (`tol = 1e-10`)
  - `omp_atomic` — OpenMP element loop protected by `#pragma omp atomic`
  - `omp_colored` — **8-color lock-free element batching** (zero shared vertices within any color,
    zero atomics, bit-identical CG convergence to `~2e-15` of serial)
- **Verification before timing**: `make test` checks `O(h²)` spatial convergence (`4.00×` L2 error
  drop when doubling `nx: 32 → 64`) and verifies `||u_omp - u_serial||_∞ < 1e-8`.

## Results

Machine: WSL2 Ubuntu 24.04, `Intel Core i7-14650HX` (**6 physical cores / 12 threads**), gcc 16.2.0
(`-O3 -march=native -fopenmp`). Best of 5 (`nx=400`) or 3 (`nx=800`) timed runs after an untimed warmup.
Effective CSR SpMV bandwidth uses compulsory traffic `(12·nnz + 16·dof)` bytes per iteration.

### 1. Cache-resident mesh (`nx = 400`, `159,201` DOFs, `1,111,217` NNZ, ~16 MB, 571 CG iters, `L2 = 2.614e-6`)

| mode | threads | total (ms) | speedup | assembly (ms) | SpMV (ms) | vec ops (ms) | SpMV BW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `serial` | 1 | 486.0 | 1.00× | 17.67 | 333.3 | 134.3 | 27.2 GB/s |
| `omp_colored` | 2 | 240.2 | 2.02× | 10.30 | 168.6 | 60.7 | 53.8 GB/s |
| `omp_colored` | 4 | 127.4 | 3.82× | 7.02 | 87.8 | 31.9 | 103.2 GB/s |
| `omp_colored` | 6 | 131.5 | 3.70× | 5.89 | 96.0 | 29.1 | 94.4 GB/s |
| `omp_atomic` | 12 | 85.1 | 5.71× | 5.19 | 58.9 | 20.4 | 154.0 GB/s |
| `omp_colored` | 12 | **82.2** | **5.92×** | **3.89 (4.55×)** | **57.5** | **20.2** | **157.8 GB/s** |

### 2. DRAM-bound mesh (`nx = 800`, `638,401` DOFs, `4,462,417` NNZ, ~64 MB past L3, 1,098 CG iters, `L2 = 6.535e-7`)

| mode | threads | total (s) | speedup | assembly (ms) | SpMV (s) | vec ops (s) | SpMV BW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `serial` | 1 | 4.729 | 1.00× | 66.54 | 3.173 | 1.486 | 22.1 GB/s |
| `omp_colored` | 2 | 2.699 | 1.75× | 39.73 | 1.851 | 0.806 | 37.8 GB/s |
| `omp_colored` | 4 | 1.666 | 2.84× | 24.69 | 1.232 | 0.408 | 56.8 GB/s |
| `omp_colored` | 6 | 1.657 | 2.85× | 18.02 | 1.250 | 0.388 | 56.0 GB/s |
| `omp_atomic` | 12 | 1.475 | 3.21× | 20.05 | 1.148 | 0.306 | 61.0 GB/s |
| `omp_colored` | 12 | **1.452** | **3.26×** | **15.92 (4.18×)** | **1.132** | **0.303** | **61.9 GB/s** |

## GPU: same matrix, different schedule

`fem_solver -n <nx> --export build/m<nx>.csr` writes the assembled CSR, and `src/cg_cuda_dev.cu` solves the
same system on the GPU. Two GPU versions exist because the first one had a lesson in it.

| case | iters | CPU 12 threads | GPU v1: host syncs | GPU v2: device loop | v2 vs CPU |
| --- | --- | --- | --- | --- | --- |
| nx=400, 159k DOF | 571 / 580 | 65 ms | 147 ms (**0.44×**) | **56 ms** | **1.15×** |
| nx=800, 638k DOF | 1098 / 1100 | 962 ms | 465 ms (2.07×) | **331 ms** | **2.91×** |

Both GPU versions converge to the same solution as the CPU run (`sum(x) = 64844.557535290` at 159k,
`259381.230124981` at 638k; relative residual 7.5e-11 and 2.2e-10).

- **v1** synchronised with the host three times per iteration: three dot products, each a device→host copy.
  At 159k DOF one SpMV is only 19 µs, so those round trips were ~93% of the iteration — the GPU lost to a
  12-thread CPU.
- **v2** keeps the reductions and the scalars `alpha` / `beta` on the device. The two dots needed per
  iteration are reduced in place, `alpha` and `beta` come from a one-thread kernel, and the `axpy` / `axpby`
  kernels read their scalar from device memory. The only host read left is the residual check.
- **The maths never changed.** Same matrix, same tolerance, same iteration count. The schedule changed, and
  the small case went from 0.44× to 1.15×.

**How often to check convergence** (159k DOF, tolerance 1e-10):

| residual check every | iterations | solve time | ms/iter |
| --- | --- | --- | --- |
| 1 | 571 | 77 ms | 0.136 |
| 5 | 575 | 61 ms | 0.106 |
| **20** | 580 | **56 ms** | **0.097** |
| 50 | 600 | 65 ms | 0.109 |
| 100 | 600 | 58 ms | 0.097 |

Checking every iteration costs 37% more than every 20; checking every 50 overshoots to 600 iterations.
The sweet spot is a property of the machine and the matrix, which is why it is a flag (`sync_every`).

**Burst is not sustained.** The same SpMV kernel, same matrix, measured at three repetition counts:

| repetitions | ms per SpMV | GB/s |
| --- | --- | --- |
| 30 | 0.162 | 392.9 |
| 1,098 | 0.180 | 354.4 |
| 3,000 | 0.331 | **192.8** |

A laptop GPU does not hold its burst bandwidth. Every number in this README states its run length for that
reason.

Reproduce: `make gpu-bench` (kernels) and `make cg-both` (both CG versions).

## Three takeaways for CAE performance engineering

1. **Why optimize SpMV before assembly:** On `nx=800` at 12 threads, CSR assembly takes **1.1%**
   (`15.9 ms`) of wall time, whereas CSR SpMV (**77.9%**, `1.132 s`) and CG vector ops (**20.9%**,
   `0.303 s`) consume **98.8%**.
2. **Graph coloring beats atomics by ~21–25% on assembly, without changing solver math:**
   `omp_colored` cuts assembly from `5.19 → 3.89 ms` (`nx=400`) and `20.05 → 15.92 ms` (`nx=800`)
   while maintaining `O(h²)` accuracy (`4.00×` L2 drop from `nx=400` to `800`) and `~2e-15` agreement.
3. **Crossing L3 exposes the exact DRAM wall from `hpc-kernels`:** At `nx=400` (~16 MB), SpMV
   scales inside L3 to **157.8 GB/s** (`5.92×` total speedup). At `nx=800` (~64 MB), SpMV flattens
   at **56.8 → 61.9 GB/s** from 4 to 12 threads — matching the **58.35 GB/s** `axpy` DRAM ceiling
   measured independently in [hpc-kernels](https://github.com/cccjj55/hpc-kernels). Beyond 4 threads,
   adding cores yields <15% total gain; cutting CG iterations (preconditioning / multigrid) or moving
   matrix-free is what actually moves the needle.

## Build and run

```bash
make build
make test                           # O(h^2) convergence + OpenMP equivalence check
./build/fem_solver -n 400 -r 5 --csv > results/fem_400.csv
./build/fem_solver -n 800 -r 3 --csv > results/fem_800.csv
python3 scripts/plot_svg.py results/fem_400.csv results/fem_800.csv docs/fem_breakdown.svg
```

## License

MIT
