#include "attention.cu"
#include <cmath>
#include <math_constants.h>

// Educational fused baseline.
// One block handles one query row.
// Requires n <= 256 and d <= 256 for this version.
__global__ void attention_fused_row(
    const float* q, const float* k, const float* v,
    float* o, int n, int d, float scale) {
  int i = blockIdx.x;
  int t = threadIdx.x;

  __shared__ float red[256];
  __shared__ float m;
  __shared__ float denom;

  float score = -CUDART_INF_F;

  if (t < n && t <= i) {
    score = 0.0f;
    for (int x = 0; x < d; ++x) {
      score += q[i * d + x] * k[t * d + x];
    }
    score *= scale;
  }

  red[t] = score;
  __syncthreads();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (t < stride) {
      red[t] = fmaxf(red[t], red[t + stride]);
    }
    __syncthreads();
  }

  if (t == 0) m = red[0];
  __syncthreads();

  float w = (t < n && t <= i) ? expf(score - m) : 0.0f;
  red[t] = w;
  __syncthreads();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (t < stride) {
      red[t] += red[t + stride];
    }
    __syncthreads();
  }

  if (t == 0) denom = red[0];
  __syncthreads();

  if (t < d) {
    float z = 0.0f;
    for (int j = 0; j <= i; ++j) {
      float a = 0.0f;
      for (int x = 0; x < d; ++x) {
        a += q[i * d + x] * k[j * d + x];
      }
      z += expf(a * scale - m) / denom * v[j * d + t];
    }
    o[i * d + t] = z;
  }
}

void launch_fused_attention(
    const float* q, const float* k, const float* v,
    float* o, int n, int d, float scale) {
  attention_fused_row<<<n, 256>>>(q, k, v, o, n, d, scale);
}