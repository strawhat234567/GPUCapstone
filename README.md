# Tiled Causal Attention in Plain CUDA

This project implements and benchmarks three CUDA kernels for causal self-attention on a single attention head: a naive baseline, a shared-memory tiled kernel, and an online-softmax tiled kernel. Its purpose is to show how kernel structure on the GPU changes memory usage and runtime performance while preserving numerical correctness, using the same attention operation expressed in three designs. In the naive and shared-memory tiled implementations, the code explicitly forms the attention score matrix and then applies softmax and value weighting, whereas the online-softmax tiled implementation follows the FlashAttention idea of processing one key/value tile at a time and updating the softmax normalization incrementally so the full score matrix is never stored in global memory. The project draws on ThunderKittens for a tile-oriented view of AI kernels and on online-softmax formulations used to derive FlashAttention, adapting those concepts to a plain-CUDA setting suitable for a teaching benchmark.

A central goal is to understand why the kernels behave differently, not just to demonstrate that they run. To support that goal, the repository includes a CPU reference implementation for validation, timing with CUDA events, and experiments across multiple sequence lengths and head dimensions, so both numerical error and runtime can be examined side by side.

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