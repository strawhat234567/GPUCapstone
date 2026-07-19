#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do {                                         \
  cudaError_t e = (call);                                             \
  if (e != cudaSuccess) {                                             \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
            __FILE__, __LINE__, cudaGetErrorString(e));               \
    exit(1);                                                          \
  }                                                                   \
} while (0)

constexpr int TILE = 16;
constexpr int MAX_D = 128;
constexpr int FUSED_THREADS = 128;

void launch_naive_attention(
    const float* q, const float* k, const float* v,
    float* s, float* o, int n, int d, float scale);

void launch_tiled_attention(
    const float* q, const float* k, const float* v,
    float* s, float* o, int n, int d, float scale);

void launch_online_tiled_attention(
    const float* q, const float* k, const float* v,
    float* o, int n, int d, float scale);

void cpu_attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>& o,
    int n,
    int d);