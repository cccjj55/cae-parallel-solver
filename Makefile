CC ?= gcc
CFLAGS ?= -O3 -march=native -fopenmp -Wall -Wextra -std=c11
LDFLAGS ?= -lm

BIN := build/fem_solver

.PHONY: all build test bench clean

all: build

build: $(BIN)

$(BIN): src/fem_solver.c
	@mkdir -p build
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test: build
	./$(BIN) --test

bench: build
	@mkdir -p results
	./$(BIN) -n 400 -r 3 --csv > results/fem.csv
	cat results/fem.csv

clean:
	rm -rf build results

# GPU path is optional: needs nvcc and a CUDA-capable device.
NVCC ?= nvcc
CUDA_ARCH ?= sm_120
HOSTCXX ?= g++

cuda:
	@mkdir -p build
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -ccbin $(HOSTCXX) src/spmv_cuda.cu -o build/spmv_cuda

# Same matrix on both sides: export the CPU-assembled CSR, then feed it to the kernels.
gpu-bench: build cuda
	./build/fem_solver -n 400 --export build/m400.csr
	./build/fem_solver -n 800 --export build/m800.csr
	./build/spmv_cuda build/m400.csr 30
	./build/spmv_cuda build/m800.csr 30
