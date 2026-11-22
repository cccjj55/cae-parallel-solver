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
