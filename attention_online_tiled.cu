#include "attention.cu"
#include <cmath>
#include <math_constants.h>


// One block computes one query row.
// Threads cooperate over one K/V tile at a time.
// Thread 0 computes the scalar recurrence state (m, l).
// All threads with tid < d update slices of the output vector.
// Educational version: simple and structurally correct online-softmax tiling.
__global__ void attention_online_tiled_kernel(
    const float* q, const float* k, const float* v,
    float* o, int n, int d, float scale) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  __shared__ float q_row[MAX_D];
  __shared__ float k_tile[TILE][MAX_D];
  __shared__ float v_tile[TILE][MAX_D];
  __shared__ float scores[TILE];
  __shared__ float weights[TILE];
  __shared__ float tile_max;
  __shared__ float old_m;
  __shared__ float new_m;
  __shared__ float old_l;
  __shared__ float new_l;
  __shared__ float out_acc[MAX_D];

  if (tid < d) {
    q_row[tid] = q[row * d + tid];
    out_acc[tid] = 0.0f;
  }
  if (tid == 0) {
    old_m = -CUDART_INF_F;
    old_l = 0.0f;
  }
  __syncthreads();

  int last_col = row;

  for (int tile_start = 0; tile_start <= last_col; tile_start += TILE) {
    int valid = last_col - tile_start + 1;
    if (valid > TILE) valid = TILE;

    for (int idx = tid; idx < valid * d; idx += blockDim.x) {
      int r = idx / d;
      int c = idx % d;
      k_tile[r][c] = k[(tile_start + r) * d + c];
      v_tile[r][c] = v[(tile_start + r) * d + c];
    }
    __syncthreads();

    if (tid < valid) {
      float s = 0.0f;
      for (int x = 0; x < d; ++x) {
        s += q_row[x] * k_tile[tid][x];
      }
      scores[tid] = s * scale;
    }
    __syncthreads();

    if (tid == 0) {
      float local_max = -CUDART_INF_F;
      for (int j = 0; j < valid; ++j) {
        local_max = fmaxf(local_max, scores[j]);
      }

      tile_max = local_max;
      new_m = fmaxf(old_m, tile_max);

      float tile_sum = 0.0f;
      for (int j = 0; j < valid; ++j) {
        weights[j] = expf(scores[j] - new_m);
        tile_sum += weights[j];
      }

      new_l = old_l * expf(old_m - new_m) + tile_sum;
    }
    __syncthreads();

    if (tid < d) {
      float pv = 0.0f;
      for (int j = 0; j < valid; ++j) {
        pv += weights[j] * v_tile[j][tid];
      }

      float old_term = (old_l == 0.0f) ? 0.0f : (old_l * expf(old_m - new_m) / new_l) * out_acc[tid];
      float new_term = pv / new_l;
      out_acc[tid] = old_term + new_term;
    }
    __syncthreads();

    if (tid == 0) {
      old_m = new_m;
      old_l = new_l;
    }
    __syncthreads();
  }

  if (tid < d) {
    o[row * d + tid] = out_acc[tid];
  }
}

void launch_online_tiled_attention(
    const float* q, const float* k, const float* v,
    float* o, int n, int d, float scale) {
  attention_online_tiled_kernel<<<n, FUSED_THREADS>>>(q, k, v, o, n, d, scale);
}