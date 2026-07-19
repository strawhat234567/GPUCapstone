#include "attention.cu"
#include <cmath>
#include <math_constants.h>

__global__ void scores_naive(
    const float* q, const float* k, float* s,
    int n, int d, float scale) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= n || j >= n) return;

  if (j > i) {
    s[i * n + j] = -CUDART_INF_F;
    return;
  }

  float dot = 0.0f;
  for (int x = 0; x < d; ++x) {
    dot += q[i * d + x] * k[j * d + x];
  }
  s[i * n + j] = dot * scale;
}

__global__ void row_softmax(float* s, int n) {
  int i = blockIdx.x;
  extern __shared__ float tmp[];

  float local = -CUDART_INF_F;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    local = fmaxf(local, s[i * n + j]);
  }
  tmp[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      tmp[threadIdx.x] = fmaxf(tmp[threadIdx.x], tmp[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  float m = tmp[0];
  float sum = 0.0f;

  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    float e = expf(s[i * n + j] - m);
    s[i * n + j] = e;
    sum += e;
  }
  tmp[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      tmp[threadIdx.x] += tmp[threadIdx.x + stride];
    }
    __syncthreads();
  }

  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    s[i * n + j] /= tmp[0];
  }
}

__global__ void weighted_value(
    const float* p, const float* v, float* o,
    int n, int d) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y;

  if (i >= n || x >= d) return;

  float acc = 0.0f;
  for (int j = 0; j <= i; ++j) {
    acc += p[i * n + j] * v[j * d + x];
  }
  o[i * d + x] = acc;
}

void launch_naive_attention(
    const float* q, const float* k, const float* v,
    float* s, float* o, int n, int d, float scale) {
  dim3 block2(16, 16);
  dim3 grid2((n + 15) / 16, (n + 15) / 16);
  dim3 block_out(256);
  dim3 grid_out((d + 255) / 256, n);

  scores_naive<<<grid2, block2>>>(q, k, s, n, d, scale);
  row_softmax<<<n, 256, 256 * sizeof(float)>>>(s, n);
  weighted_value<<<grid_out, block_out>>>(s, v, o, n, d);
}