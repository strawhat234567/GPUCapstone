#include "attention.cu"
#include <cmath>
#include <math_constants.h>


__global__ void scores_tiled(
    const float* q, const float* k, float* s,
    int n, int d, float scale) {
  __shared__ float qs[TILE][MAX_D];
  __shared__ float ks[TILE][MAX_D];

  int qi = blockIdx.y * TILE + threadIdx.y;
  int kj = blockIdx.x * TILE + threadIdx.x;
  int tid = threadIdx.y * TILE + threadIdx.x;

  for (int x = tid; x < TILE * d; x += TILE * TILE) {
    int r = x / d;
    int c = x % d;

    qs[r][c] = (blockIdx.y * TILE + r < n)
        ? q[(blockIdx.y * TILE + r) * d + c]
        : 0.0f;

    ks[r][c] = (blockIdx.x * TILE + r < n)
        ? k[(blockIdx.x * TILE + r) * d + c]
        : 0.0f;
  }

  __syncthreads();

  if (qi >= n || kj >= n) return;

  if (kj > qi) {
    s[qi * n + kj] = -CUDART_INF_F;
    return;
  }

  float dot = 0.0f;
  #pragma unroll 4
  for (int x = 0; x < d; ++x) {
    dot += qs[threadIdx.y][x] * ks[threadIdx.x][x];
  }

  s[qi * n + kj] = dot * scale;
}

__global__ void row_softmax_tiled(float* s, int n) {
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

__global__ void weighted_value_tiled(
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

void launch_tiled_attention(
    const float* q, const float* k, const float* v,
    float* s, float* o, int n, int d, float scale) {
  dim3 block2(16, 16);
  dim3 grid2((n + 15) / 16, (n + 15) / 16);
  dim3 block_out(256);
  dim3 grid_out((d + 255) / 256, n);

  scores_tiled<<<grid2, block2>>>(q, k, s, n, d, scale);
  row_softmax_tiled<<<n, 256, 256 * sizeof(float)>>>(s, n);
  weighted_value_tiled<<<grid_out, block_out>>>(s, v, o, n, d);
}