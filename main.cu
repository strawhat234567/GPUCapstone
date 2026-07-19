#include "attention.cu"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

static float elapsed(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

int main(int argc, char** argv) {
  int n = 128;
  int d = 64;
  int iters = 100;
  std::string mode = "all";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--n") n = std::atoi(argv[++i]);
    else if (a == "--d") d = std::atoi(argv[++i]);
    else if (a == "--iters") iters = std::atoi(argv[++i]);
    else if (a == "--mode") mode = argv[++i];
  }

  if (d < 1 || d > MAX_D || n < 1) {
    std::cerr << "Require 1 <= d <= " << MAX_D << " and n >= 1\n";
    return 1;
  }

  size_t qkv_bytes = static_cast<size_t>(n) * d * sizeof(float);
  size_t s_bytes = static_cast<size_t>(n) * n * sizeof(float);

  std::vector<float> hq(n * d), hk(n * d), hv(n * d);
  std::vector<float> href(n * d), hout(n * d);

  std::mt19937 gen(7);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (float& x : hq) x = dist(gen);
  for (float& x : hk) x = dist(gen);
  for (float& x : hv) x = dist(gen);

  float *q, *k, *v, *s, *o;
  CUDA_CHECK(cudaMalloc(&q, qkv_bytes));
  CUDA_CHECK(cudaMalloc(&k, qkv_bytes));
  CUDA_CHECK(cudaMalloc(&v, qkv_bytes));
  CUDA_CHECK(cudaMalloc(&s, s_bytes));
  CUDA_CHECK(cudaMalloc(&o, qkv_bytes));

  CUDA_CHECK(cudaMemcpy(q, hq.data(), qkv_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(k, hk.data(), qkv_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(v, hv.data(), qkv_bytes, cudaMemcpyHostToDevice));

  cpu_attention(hq, hk, hv, href, n, d);
  float scale = 1.0f / std::sqrt(static_cast<float>(d));

  cudaEvent_t st, en;
  CUDA_CHECK(cudaEventCreate(&st));
  CUDA_CHECK(cudaEventCreate(&en));

  auto report = [&](const char* name, auto launch) {
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(st));
    for (int z = 0; z < iters; ++z) {
      launch();
    }
    CUDA_CHECK(cudaEventRecord(en));
    CUDA_CHECK(cudaEventSynchronize(en));

    CUDA_CHECK(cudaMemcpy(hout.data(), o, qkv_bytes, cudaMemcpyDeviceToHost));

    float err = 0.0f;
    for (size_t z = 0; z < hout.size(); ++z) {
      err = std::max(err, std::fabs(hout[z] - href[z]));
    }

    std::cout << name << ": "
              << elapsed(st, en) / iters
              << " ms, max_abs_error=" << err << "\n";
  };

  if (mode == "naive" || mode == "all") {
    report("naive", [&]() {
      launch_naive_attention(q, k, v, s, o, n, d, scale);
    });
  }

  if (mode == "tiled" || mode == "all") {
    report("shared_tiled", [&]() {
      launch_tiled_attention(q, k, v, s, o, n, d, scale);
    });
  }

  if (mode == "online" || mode == "all") {
    report("online_tiled", [&]() {
      launch_online_tiled_attention(q, k, v, o, n, d, scale);
    });
  }

  CUDA_CHECK(cudaFree(q));
  CUDA_CHECK(cudaFree(k));
  CUDA_CHECK(cudaFree(v));
  CUDA_CHECK(cudaFree(s));
  CUDA_CHECK(cudaFree(o));
  CUDA_CHECK(cudaEventDestroy(st));
  CUDA_CHECK(cudaEventDestroy(en));

  return 0;
}