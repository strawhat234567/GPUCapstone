# Tiled Causal Attention in Plain CUDA

This project moves past a basic CUDA demonstration by implementing and validating several kernel designs for the same nontrivial linear-algebra workload, measuring their behavior empirically, and including a fused attention kernel based on online softmax rather than a superficial parallelization of the standard score-and-softmax steps.

## Implementations

- `naive`: Materializes the causal score matrix, applies row-wise softmax in a separate stage, and then computes `P V`.
- `shared_tiled`: Loads `16 x D` tiles of `Q` and `K` into shared memory during score computation.
- `fused_no_score_matrix`: A teaching baseline that never writes the `N x N` score matrix to global memory. It currently requires `N <= 256` and recomputes scores while forming the output. That design makes the memory tradeoff easy to inspect, but it does not guarantee higher speed.

## Build and run

```bash
make
./attention_bench --n 128 --d 64 --iters 100
```

Run a single implementation:

```bash
./attention_bench --mode naive --n 256 --d 64 --iters 100
./attention_bench --mode tiled --n 256 --d 64 --iters 100
./attention_bench --mode fused --n 128 --d 64 --iters 100
```

The fused kernel requires `N <= 256`. The baseline kernels support larger `N`, subject to available GPU memory.

## Suggested experiments

- Fix `D = 64` and sweep `N = 64, 128, 256, 512` for the baseline kernels.
- Compare `naive` and `shared_tiled`, then evaluate whether shared-memory staging helps on the target GPU.
- Sweep `D = 32, 64, 128`. The tiled kernel supports `D <= 128`.
- Examine why the fused kernel removes an `N x N` global-memory allocation but recomputes dot products. That tradeoff sets up the next step: an online-softmax tiled kernel.

## Next milestone

Replace `attention_fused_row` with a block-tiled online-softmax kernel. Maintain each query row's running maximum `m` and normalization term `l` across key tiles, then accumulate the output vector tile by tile. This is the conceptual bridge to FlashAttention-style kernels.

## CUDA environment check

```bash
nvidia-smi
nvcc --version
```

If `nvcc` is unavailable, compile through the Coursera notebook's CUDA cell or adapt the single `main.cu` source to the notebook's expected build command.

## references

https://github.com/HazyResearch/train-tk
https://hazyresearch.stanford.edu/blog/2024-05-12-tk
https://arxiv.org/html/2410.00907v1