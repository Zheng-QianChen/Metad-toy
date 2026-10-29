/*
 * Standalone performance/accuracy probe (assessment prototype; not production).
 *
 * Compile / run (expected: -O3, no fast-math; C++17):
 *   nvcc -O3 -std=c++17 -o /tmp/bias_benchmark benchmarks/highdim/evaluate_benchmark.cu
 *   /tmp/bias_benchmark
 *
 * Compares:
 *   - Gaussian HILLS energy+gradient: runtime-D vs templated-D (1,2,3,8,32,128) on CPU
 *   - Catmull-Rom tensor grid interp energy+gradient: runtime vs templated D=1/2/3 (same data)
 *   - Resident-GPU HILLS vs CPU scalar; GPU end-to-end per-query (coords+V/grad+stream sync,
 *     transfer included); batched queries; upload/rebuild cost reported separately
 *
 * Machine-readable JSON lines on stdout (kind, D, H, block, queries, time_us, max_abs_error, passed, …).
 * Device inventory via CUDA runtime; unavailable GPU is reported and GPU suites skipped (exit 0).
 * Bound intended for <60s wall on RTX 4080-class hardware.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kAbsTol = 1e-9;
constexpr int kMedianSamples = 5;
constexpr int kWarmup = 2;

struct CudaBuffer {
  void* p = nullptr;
  size_t bytes = 0;
  CudaBuffer() = default;
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
  CudaBuffer(CudaBuffer&& o) noexcept : p(o.p), bytes(o.bytes) {
    o.p = nullptr;
    o.bytes = 0;
  }
  ~CudaBuffer() { reset(); }
  void reset() {
    if (p) {
      cudaFree(p);
      p = nullptr;
      bytes = 0;
    }
  }
  cudaError_t alloc(size_t n) {
    reset();
    bytes = n;
    if (n == 0) return cudaSuccess;
    return cudaMalloc(&p, n);
  }
  template <typename T>
  T* as() {
    return static_cast<T*>(p);
  }
};

struct CudaStreamRAII {
  cudaStream_t s = nullptr;
  explicit CudaStreamRAII(bool create) {
    if (create && cudaStreamCreate(&s) != cudaSuccess) s = nullptr;
  }
  ~CudaStreamRAII() {
    if (s) cudaStreamDestroy(s);
  }
  CudaStreamRAII(const CudaStreamRAII&) = delete;
  CudaStreamRAII& operator=(const CudaStreamRAII&) = delete;
  bool ok() const { return s != nullptr; }
};

bool cuda_ok(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  std::fprintf(stderr, "cuda error (%s): %s\n", what, cudaGetErrorString(e));
  return false;
}

double median_us(std::vector<double>& samples) {
  std::sort(samples.begin(), samples.end());
  const size_t n = samples.size();
  if (n == 0) return 0.0;
  if (n & 1u) return samples[n / 2];
  return 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

volatile double g_sink = 0.0;
inline void consume(double v) { g_sink = g_sink * 0.5 + v; }
inline void consume_vec(const double* a, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += a[i];
  consume(s);
}

void emit_json(const char* kind, int D, int H, int block, int queries, double time_us,
               double max_abs_error, int passed, const char* extra_kv = nullptr) {
  std::printf(
      "{\"kind\":\"%s\",\"D\":%d,\"H\":%d,\"block\":%d,\"queries\":%d,"
      "\"time_us\":%.6f,\"max_abs_error\":%.16e,\"passed\":%s",
      kind, D, H, block, queries, time_us, max_abs_error, passed ? "true" : "false");
  if (extra_kv && extra_kv[0]) std::printf(",%s", extra_kv);
  std::printf("}\n");
  std::fflush(stdout);
}

int iters_for(int D, int H, int queries, bool heavy) {
  const double work = static_cast<double>(D) * static_cast<double>(H) *
                      static_cast<double>(std::max(queries, 1));
  int base = heavy ? 40 : 200;
  if (work > 5e5) base = heavy ? 8 : 40;
  if (work > 2e6) base = heavy ? 3 : 12;
  if (work > 8e6) base = heavy ? 1 : 4;
  if (D >= 128 && H >= 10000) base = 1;
  return std::max(1, base);
}

// -----------------------------------------------------------------------------
// Gaussian HILLS
// -----------------------------------------------------------------------------

inline void hills_eval_runtime(int D, int H, const double* centers, const double* heights,
                               double sigma, const double* s, double* V_out, double* grad) {
  const double inv_s2 = 1.0 / (sigma * sigma);
  double V = 0.0;
  for (int d = 0; d < D; ++d) grad[d] = 0.0;
  for (int h = 0; h < H; ++h) {
    double r2 = 0.0;
    const double* c = centers + static_cast<size_t>(h) * static_cast<size_t>(D);
    for (int d = 0; d < D; ++d) {
      const double dx = s[d] - c[d];
      r2 += dx * dx;
    }
    const double g = heights[h] * std::exp(-0.5 * r2 * inv_s2);
    V += g;
    for (int d = 0; d < D; ++d) {
      const double dx = s[d] - c[d];
      grad[d] += g * (-dx * inv_s2);
    }
  }
  *V_out = V;
}

template <int D>
inline void hills_eval_templated(int H, const double* centers, const double* heights, double sigma,
                                 const double* s, double* V_out, double* grad) {
  const double inv_s2 = 1.0 / (sigma * sigma);
  double V = 0.0;
#pragma unroll
  for (int d = 0; d < D; ++d) grad[d] = 0.0;
  for (int h = 0; h < H; ++h) {
    double r2 = 0.0;
    const double* c = centers + static_cast<size_t>(h) * static_cast<size_t>(D);
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const double dx = s[d] - c[d];
      r2 += dx * dx;
    }
    const double g = heights[h] * std::exp(-0.5 * r2 * inv_s2);
    V += g;
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const double dx = s[d] - c[d];
      grad[d] += g * (-dx * inv_s2);
    }
  }
  *V_out = V;
}

void hills_eval_templated_dispatch(int D, int H, const double* centers, const double* heights,
                                   double sigma, const double* s, double* V_out, double* grad) {
  switch (D) {
    case 1:
      hills_eval_templated<1>(H, centers, heights, sigma, s, V_out, grad);
      break;
    case 2:
      hills_eval_templated<2>(H, centers, heights, sigma, s, V_out, grad);
      break;
    case 3:
      hills_eval_templated<3>(H, centers, heights, sigma, s, V_out, grad);
      break;
    case 8:
      hills_eval_templated<8>(H, centers, heights, sigma, s, V_out, grad);
      break;
    case 32:
      hills_eval_templated<32>(H, centers, heights, sigma, s, V_out, grad);
      break;
    case 128:
      hills_eval_templated<128>(H, centers, heights, sigma, s, V_out, grad);
      break;
    default:
      hills_eval_runtime(D, H, centers, heights, sigma, s, V_out, grad);
      break;
  }
}

double max_abs_diff(const double* a, const double* b, int n) {
  double m = 0.0;
  for (int i = 0; i < n; ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

// -----------------------------------------------------------------------------
// Catmull-Rom dense tensor grid (D=1,2,3)
// -----------------------------------------------------------------------------

inline double cr_value(double p0, double p1, double p2, double p3, double x) {
  return p1 + 0.5 * x *
                 ((p2 - p0) + x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) +
                  x * x * (-p0 + 3.0 * p1 - 3.0 * p2 + p3));
}

inline double cr_deriv(double p0, double p1, double p2, double p3, double x) {
  return 0.5 * ((p2 - p0) + 2.0 * x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) +
                3.0 * x * x * (-p0 + 3.0 * p1 - 3.0 * p2 + p3));
}

inline void center_frac_clamp(double cv, double lo, double h, int nb, int& i, double& f) {
  double u = (cv - lo) / h - 0.5;
  i = static_cast<int>(std::floor(u));
  f = u - i;
  if (i < 1) {
    i = 1;
    f = 0.0;
  }
  if (i > nb - 3) {
    i = nb - 3;
    f = 1.0;
  }
}

struct DenseGrid {
  int D = 0;
  int nbin[3]{};
  double lo[3]{};
  double h[3]{};
  std::vector<double> data;

  double at(const int* idx) const {
    if (D == 1) return data[static_cast<size_t>(idx[0])];
    if (D == 2)
      return data[static_cast<size_t>(idx[0]) * static_cast<size_t>(nbin[1]) +
                  static_cast<size_t>(idx[1])];
    return data[static_cast<size_t>(idx[0]) * static_cast<size_t>(nbin[1]) *
                    static_cast<size_t>(nbin[2]) +
                static_cast<size_t>(idx[1]) * static_cast<size_t>(nbin[2]) +
                static_cast<size_t>(idx[2])];
  }
};

double grid_mixed(const DenseGrid& G, int current, int target, int* coord, const double* frac) {
  if (current == G.D) return G.at(coord);
  double p[4];
  const int origin = coord[current];
  const double x = frac[current];
  for (int t = -1; t <= 2; ++t) {
    int c = origin + t;
    if (c < 0) c = 0;
    if (c >= G.nbin[current]) c = G.nbin[current] - 1;
    coord[current] = c;
    p[t + 1] = grid_mixed(G, current + 1, target, coord, frac);
  }
  coord[current] = origin;
  if (current == target) return cr_deriv(p[0], p[1], p[2], p[3], x) / G.h[current];
  return cr_value(p[0], p[1], p[2], p[3], x);
}

void grid_eval_runtime(const DenseGrid& G, const double* cv, double* V_out, double* grad) {
  int base[3]{};
  double frac[3]{};
  for (int d = 0; d < G.D; ++d) center_frac_clamp(cv[d], G.lo[d], G.h[d], G.nbin[d], base[d], frac[d]);
  int coord[3] = {base[0], base[1], base[2]};
  *V_out = grid_mixed(G, 0, -1, coord, frac);
  for (int k = 0; k < G.D; ++k) {
    coord[0] = base[0];
    coord[1] = base[1];
    coord[2] = base[2];
    grad[k] = grid_mixed(G, 0, k, coord, frac);
  }
}

// 两条路径使用相同递归与重复求导算法，仅固定 D，隔离模板本身的收益。
template<int D>
double grid_mixed_static(const DenseGrid& grid, int current, int target,
                         int* coordinates, const double* fraction) {
  if (current == D) {
    size_t offset=0;
    for (int axis=0; axis<D; ++axis) offset=offset*grid.nbin[axis]+coordinates[axis];
    return grid.data[offset];
  }
  double values[4];
  const int origin=coordinates[current];
  for (int stencil=-1; stencil<=2; ++stencil) {
    coordinates[current]=std::max(0,std::min(origin+stencil,grid.nbin[current]-1));
    values[stencil+1]=grid_mixed_static<D>(grid,current+1,target,coordinates,fraction);
  }
  coordinates[current]=origin;
  return current==target
    ? cr_deriv(values[0],values[1],values[2],values[3],fraction[current])/grid.h[current]
    : cr_value(values[0],values[1],values[2],values[3],fraction[current]);
}
template<int D>
void grid_eval_templated(const DenseGrid& grid,const double* point,double* energy,double* gradient) {
  int coordinates[3]={};
  double fraction[3]={};
  for(int axis=0;axis<D;++axis)
    center_frac_clamp(point[axis],grid.lo[axis],grid.h[axis],grid.nbin[axis],coordinates[axis],fraction[axis]);
  *energy=grid_mixed_static<D>(grid,0,-1,coordinates,fraction);
  for(int axis=0;axis<D;++axis)
    gradient[axis]=grid_mixed_static<D>(grid,0,axis,coordinates,fraction);
}

DenseGrid make_grid(int D, std::mt19937_64& rng) {
  DenseGrid G;
  G.D = D;
  // Modest bins so rebuild stays cheap; enough for valid Catmull-Rom stencil.
  const int nb = (D == 1) ? 64 : (D == 2) ? 32 : 16;
  for (int d = 0; d < D; ++d) {
    G.nbin[d] = nb;
    G.lo[d] = 0.0;
    G.h[d] = 1.0 / static_cast<double>(nb);
  }
  size_t n = 1;
  for (int d = 0; d < D; ++d) n *= static_cast<size_t>(G.nbin[d]);
  G.data.resize(n);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  for (size_t i = 0; i < n; ++i) G.data[i] = U(rng);
  return G;
}

void fill_hills(int D, int H, std::vector<double>& centers, std::vector<double>& heights,
                std::vector<double>& query, double& sigma, std::mt19937_64& rng) {
  centers.resize(static_cast<size_t>(H) * static_cast<size_t>(D));
  heights.resize(static_cast<size_t>(H));
  query.resize(static_cast<size_t>(D));
  std::uniform_real_distribution<double> Uc(-0.5, 0.5);
  std::uniform_real_distribution<double> Uh(0.1, 1.0);
  for (int h = 0; h < H; ++h) {
    heights[static_cast<size_t>(h)] = Uh(rng);
    for (int d = 0; d < D; ++d) centers[static_cast<size_t>(h) * D + d] = Uc(rng);
  }
  for (int d = 0; d < D; ++d) query[static_cast<size_t>(d)] = Uc(rng);
  sigma = 0.35 * std::sqrt(double(D)); // 保持高维能量/梯度有可测信号。
}

// -----------------------------------------------------------------------------
// GPU kernels: per-hill contribution, deterministic block reduce, optional 2nd pass
// Max D supported on device path for this probe.
// -----------------------------------------------------------------------------

constexpr int kGpuMaxD = 128;

__device__ void block_reduce_add(double* sh, int n, int tid) {
  // Deterministic tree; n must be a power of two (pad unused lanes with 0).
  for (int stride = n >> 1; stride > 0; stride >>= 1) {
    __syncthreads();
    if (tid < stride) sh[tid] += sh[tid + stride];
  }
  __syncthreads();
}

inline int next_pow2_host(int v) {
  int p = 1;
  while (p < v) p <<= 1;
  return p;
}

// Shared mem = one lane of blockDim doubles; reduce V then each grad dim sequentially
// so D=128 / block=256 stays within default shared-memory limits.

__global__ void hills_partial_kernel(int D, int H, const double* __restrict__ centers,
                                     const double* __restrict__ heights, double sigma,
                                     const double* __restrict__ query, double* __restrict__ block_V,
                                     double* __restrict__ block_grad) {
  extern __shared__ double sh[];
  const int tid = threadIdx.x;
  const int hill = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
  const double inv_s2 = 1.0 / (sigma * sigma);

  double local_V = 0.0;
  double local_g[kGpuMaxD];
  for (int d = 0; d < D; ++d) local_g[d] = 0.0;

  if (hill < H) {
    const double* c = centers + static_cast<size_t>(hill) * static_cast<size_t>(D);
    double r2 = 0.0;
    for (int d = 0; d < D; ++d) {
      const double dx = query[d] - c[d];
      r2 += dx * dx;
    }
    local_V = heights[hill] * exp(-0.5 * r2 * inv_s2);
    for (int d = 0; d < D; ++d) {
      const double dx = query[d] - c[d];
      local_g[d] = local_V * (-dx * inv_s2);
    }
  }

  sh[tid] = local_V;
  block_reduce_add(sh, blockDim.x, tid);
  if (tid == 0) block_V[blockIdx.x] = sh[0];

  for (int d = 0; d < D; ++d) {
    sh[tid] = local_g[d];
    block_reduce_add(sh, blockDim.x, tid);
    if (tid == 0) block_grad[static_cast<size_t>(blockIdx.x) * D + d] = sh[0];
  }
}

__global__ void hills_reduce_blocks_kernel(int D, int nBlocks, const double* __restrict__ block_V,
                                           const double* __restrict__ block_grad,
                                           double* __restrict__ V_out, double* __restrict__ grad_out) {
  extern __shared__ double sh[];
  const int tid = threadIdx.x;

  double v = 0.0;
  double g[kGpuMaxD];
  for (int d = 0; d < D; ++d) g[d] = 0.0;
  for (int i = tid; i < nBlocks; i += blockDim.x) {
    v += block_V[i];
    for (int d = 0; d < D; ++d) g[d] += block_grad[static_cast<size_t>(i) * D + d];
  }
  sh[tid] = v;
  block_reduce_add(sh, blockDim.x, tid);
  if (tid == 0) *V_out = sh[0];
  for (int d = 0; d < D; ++d) {
    sh[tid] = g[d];
    block_reduce_add(sh, blockDim.x, tid);
    if (tid == 0) grad_out[d] = sh[0];
  }
}

__global__ void hills_batched_kernel(int D, int H, int Q, const double* __restrict__ centers,
                                     const double* __restrict__ heights, double sigma,
                                     const double* __restrict__ queries, double* __restrict__ V_out,
                                     double* __restrict__ grad_out) {
  extern __shared__ double sh[];
  const int q = static_cast<int>(blockIdx.x);
  if (q >= Q) return;
  const int tid = threadIdx.x;
  const double* query = queries + static_cast<size_t>(q) * D;
  const double inv_s2 = 1.0 / (sigma * sigma);

  double local_V = 0.0;
  double local_g[kGpuMaxD];
  for (int d = 0; d < D; ++d) local_g[d] = 0.0;

  for (int h = tid; h < H; h += blockDim.x) {
    const double* c = centers + static_cast<size_t>(h) * D;
    double r2 = 0.0;
    for (int d = 0; d < D; ++d) {
      const double dx = query[d] - c[d];
      r2 += dx * dx;
    }
    const double gval = heights[h] * exp(-0.5 * r2 * inv_s2);
    local_V += gval;
    for (int d = 0; d < D; ++d) {
      const double dx = query[d] - c[d];
      local_g[d] += gval * (-dx * inv_s2);
    }
  }

  sh[tid] = local_V;
  block_reduce_add(sh, blockDim.x, tid);
  if (tid == 0) V_out[q] = sh[0];
  for (int d = 0; d < D; ++d) {
    sh[tid] = local_g[d];
    block_reduce_add(sh, blockDim.x, tid);
    if (tid == 0) grad_out[static_cast<size_t>(q) * D + d] = sh[0];
  }
}

size_t shmem_bytes(int block, int /*D*/) {
  return sizeof(double) * static_cast<size_t>(block);
}

bool launch_single_query(int D, int H, int block, const double* d_centers, const double* d_heights,
                         double sigma, const double* d_query, double* d_block_V, double* d_block_grad,
                         double* d_V, double* d_grad, cudaStream_t stream) {
  const int nBlocks = (H + block - 1) / block;
  const size_t sm = shmem_bytes(block, D);
  hills_partial_kernel<<<nBlocks, block, sm, stream>>>(D, H, d_centers, d_heights, sigma, d_query,
                                                       d_block_V, d_block_grad);
  if (cudaGetLastError() != cudaSuccess) return false;
  if (nBlocks == 1) {
    // Copy block0 results to outputs.
    if (cudaMemcpyAsync(d_V, d_block_V, sizeof(double), cudaMemcpyDeviceToDevice, stream) !=
        cudaSuccess)
      return false;
    if (cudaMemcpyAsync(d_grad, d_block_grad, sizeof(double) * static_cast<size_t>(D),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
      return false;
  } else {
    // Power-of-two threads; kernel grid-strides over nBlocks and pads extras with 0.
    int rb = next_pow2_host(std::min(nBlocks, 256));
    rb = std::max(32, std::min(rb, 256));
    hills_reduce_blocks_kernel<<<1, rb, shmem_bytes(rb, D), stream>>>(D, nBlocks, d_block_V,
                                                                      d_block_grad, d_V, d_grad);
    if (cudaGetLastError() != cudaSuccess) return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
// Benchmark suites
// -----------------------------------------------------------------------------

int run_cpu_hills() {
  const int dims[] = {1, 2, 3, 8, 32, 128};
  const int hills_list[] = {100, 1000, 10000};
  std::mt19937_64 rng(42);
  int failures = 0;

  for (int D : dims) {
    for (int H : hills_list) {
      // 128D @ 10k: fewer timed iters (iters_for handles); still accuracy-check once.
      std::vector<double> centers, heights, query, grad_rt(D), grad_tp(D);
      double sigma = 0.0, V_rt = 0.0, V_tp = 0.0;
      fill_hills(D, H, centers, heights, query, sigma, rng);

      hills_eval_runtime(D, H, centers.data(), heights.data(), sigma, query.data(), &V_rt,
                         grad_rt.data());
      hills_eval_templated_dispatch(D, H, centers.data(), heights.data(), sigma, query.data(), &V_tp,
                                    grad_tp.data());
      const double err =
          std::max(std::abs(V_rt - V_tp), max_abs_diff(grad_rt.data(), grad_tp.data(), D));
      const int pass = err <= kAbsTol;
      if (!pass) ++failures;

      const int iters = iters_for(D, H, 1, D >= 32);
      std::vector<double> samples_rt, samples_tp;
      samples_rt.reserve(kMedianSamples);
      samples_tp.reserve(kMedianSamples);

      for (int w = 0; w < kWarmup; ++w) {
        hills_eval_runtime(D, H, centers.data(), heights.data(), sigma, query.data(), &V_rt,
                           grad_rt.data());
        hills_eval_templated_dispatch(D, H, centers.data(), heights.data(), sigma, query.data(),
                                      &V_tp, grad_tp.data());
        consume(V_rt + V_tp);
        consume_vec(grad_rt.data(), D);
      }

      for (int s = 0; s < kMedianSamples; ++s) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
          asm volatile("" ::: "memory"); // 阻止编译器把相同输入求值移出计时循环。
          hills_eval_runtime(D, H, centers.data(), heights.data(), sigma, query.data(), &V_rt,
                             grad_rt.data());
          consume(V_rt);
          consume_vec(grad_rt.data(), D);
        }
        auto t1 = std::chrono::steady_clock::now();
        samples_rt.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                             static_cast<double>(iters));

        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
          asm volatile("" ::: "memory"); // 阻止编译器把相同输入求值移出计时循环。
          hills_eval_templated_dispatch(D, H, centers.data(), heights.data(), sigma, query.data(),
                                        &V_tp, grad_tp.data());
          consume(V_tp);
          consume_vec(grad_tp.data(), D);
        }
        t1 = std::chrono::steady_clock::now();
        samples_tp.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                             static_cast<double>(iters));
      }

      emit_json("cpu_hills_runtime", D, H, 0, 1, median_us(samples_rt), err, pass,
                "\"ref\":\"analytic_self\"");
      emit_json("cpu_hills_templated", D, H, 0, 1, median_us(samples_tp), err, pass,
                "\"ref\":\"analytic_self\"");
    }
  }
  return failures;
}

int run_cpu_grid() {
  const int dims[] = {1, 2, 3};
  std::mt19937_64 rng(7);
  int failures = 0;
  for (int D : dims) {
    DenseGrid G = make_grid(D, rng);
    std::vector<double> cv(D), grad_rt(D), grad_tp(D);
    std::uniform_real_distribution<double> U(0.15, 0.85);
    for (int d = 0; d < D; ++d) cv[static_cast<size_t>(d)] = U(rng);
    double V_rt = 0.0, V_tp = 0.0;
    grid_eval_runtime(G, cv.data(), &V_rt, grad_rt.data());
    if (D == 1)
      grid_eval_templated<1>(G, cv.data(), &V_tp, grad_tp.data());
    else if (D == 2)
      grid_eval_templated<2>(G, cv.data(), &V_tp, grad_tp.data());
    else
      grid_eval_templated<3>(G, cv.data(), &V_tp, grad_tp.data());
    const double err =
        std::max(std::abs(V_rt - V_tp), max_abs_diff(grad_rt.data(), grad_tp.data(), D));
    const int pass = err <= kAbsTol;
    if (!pass) ++failures;

    const int iters = 2000;
    std::vector<double> samples_rt, samples_tp;
    for (int w = 0; w < kWarmup; ++w) {
      grid_eval_runtime(G, cv.data(), &V_rt, grad_rt.data());
      if (D == 1)
        grid_eval_templated<1>(G, cv.data(), &V_tp, grad_tp.data());
      else if (D == 2)
        grid_eval_templated<2>(G, cv.data(), &V_tp, grad_tp.data());
      else
        grid_eval_templated<3>(G, cv.data(), &V_tp, grad_tp.data());
      consume(V_rt + V_tp);
    }
    for (int s = 0; s < kMedianSamples; ++s) {
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
          asm volatile("" ::: "memory"); // 阻止编译器把相同输入求值移出计时循环。
        grid_eval_runtime(G, cv.data(), &V_rt, grad_rt.data());
        consume(V_rt);
        consume_vec(grad_rt.data(), D);
      }
      auto t1 = std::chrono::steady_clock::now();
      samples_rt.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                           static_cast<double>(iters));

      t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
          asm volatile("" ::: "memory"); // 阻止编译器把相同输入求值移出计时循环。
        if (D == 1)
          grid_eval_templated<1>(G, cv.data(), &V_tp, grad_tp.data());
        else if (D == 2)
          grid_eval_templated<2>(G, cv.data(), &V_tp, grad_tp.data());
        else
          grid_eval_templated<3>(G, cv.data(), &V_tp, grad_tp.data());
        consume(V_tp);
        consume_vec(grad_tp.data(), D);
      }
      t1 = std::chrono::steady_clock::now();
      samples_tp.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                           static_cast<double>(iters));
    }
    emit_json("cpu_grid_runtime", D, 0, 0, 1, median_us(samples_rt), err, pass,
              "\"interp\":\"catmull_rom_tensor\"");
    emit_json("cpu_grid_templated", D, 0, 0, 1, median_us(samples_tp), err, pass,
              "\"interp\":\"catmull_rom_tensor\"");
  }
  return failures;
}

struct GpuHillsCtx {
  int D = 0;
  int H = 0;
  double sigma = 0.0;
  CudaBuffer centers, heights, query, block_V, block_grad, V, grad;
  CudaBuffer queries_batch, V_batch, grad_batch;
  int max_blocks = 0;
  int batch_Q = 0;

  bool upload(int D_, int H_, const std::vector<double>& h_centers, const std::vector<double>& h_heights,
              double sigma_, const std::vector<double>& h_query, int batch_Q_) {
    D = D_;
    H = H_;
    sigma = sigma_;
    batch_Q = batch_Q_;
    max_blocks = (H + 63) / 64;  // upper bound for block=64
    if (cuda_ok(centers.alloc(sizeof(double) * h_centers.size()), "alloc centers") &&
        cuda_ok(heights.alloc(sizeof(double) * h_heights.size()), "alloc heights") &&
        cuda_ok(query.alloc(sizeof(double) * h_query.size()), "alloc query") &&
        cuda_ok(block_V.alloc(sizeof(double) * static_cast<size_t>(max_blocks)), "alloc block_V") &&
        cuda_ok(block_grad.alloc(sizeof(double) * static_cast<size_t>(max_blocks) * D),
                "alloc block_grad") &&
        cuda_ok(V.alloc(sizeof(double)), "alloc V") &&
        cuda_ok(grad.alloc(sizeof(double) * static_cast<size_t>(D)), "alloc grad") &&
        cuda_ok(queries_batch.alloc(sizeof(double) * static_cast<size_t>(batch_Q) * D),
                "alloc queries_batch") &&
        cuda_ok(V_batch.alloc(sizeof(double) * static_cast<size_t>(batch_Q)), "alloc V_batch") &&
        cuda_ok(grad_batch.alloc(sizeof(double) * static_cast<size_t>(batch_Q) * D),
                "alloc grad_batch")) {
      if (!cuda_ok(cudaMemcpy(centers.as<double>(), h_centers.data(), centers.bytes,
                              cudaMemcpyHostToDevice),
                   "H2D centers"))
        return false;
      if (!cuda_ok(cudaMemcpy(heights.as<double>(), h_heights.data(), heights.bytes,
                              cudaMemcpyHostToDevice),
                   "H2D heights"))
        return false;
      if (!cuda_ok(
              cudaMemcpy(query.as<double>(), h_query.data(), query.bytes, cudaMemcpyHostToDevice),
              "H2D query"))
        return false;
      return true;
    }
    return false;
  }
};

int run_gpu_hills() {
  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e != cudaSuccess || ndev <= 0) {
    emit_json("gpu_unavailable", 0, 0, 0, 0, 0.0, 0.0, 1,
              "\"reason\":\"no_cuda_device\"");
    return 0;
  }

  for (int i = 0; i < ndev; ++i) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
    char extra[512];
    std::snprintf(extra, sizeof(extra),
                  "\"device\":%d,\"name\":\"%s\",\"major\":%d,\"minor\":%d,"
                  "\"total_global_mem_bytes\":%zu,\"multiProcessorCount\":%d",
                  i, prop.name, prop.major, prop.minor, prop.totalGlobalMem,
                  prop.multiProcessorCount);
    emit_json("device_inventory", 0, 0, 0, 0, 0.0, 0.0, 1, extra);
  }

  if (!cuda_ok(cudaSetDevice(0), "setDevice")) {
    emit_json("gpu_unavailable", 0, 0, 0, 0, 0.0, 0.0, 1, "\"reason\":\"setDevice_failed\"");
    return 0;
  }

  CudaStreamRAII stream(true);
  if (!stream.ok()) {
    emit_json("gpu_unavailable", 0, 0, 0, 0, 0.0, 0.0, 1, "\"reason\":\"stream_create_failed\"");
    return 0;
  }

  const int dims[] = {1, 2, 3, 8, 32, 128};
  const int hills_list[] = {100, 1000, 10000};
  const int blocks[] = {64, 128, 256};
  constexpr int kBatchQ = 128;
  std::mt19937_64 rng(99);
  int failures = 0;

  for (int D : dims) {
    if (D > kGpuMaxD) continue;
    for (int H : hills_list) {
      std::vector<double> centers, heights, query, grad_cpu(D), grad_gpu(D);
      double sigma = 0.0, V_cpu = 0.0, V_gpu = 0.0;
      fill_hills(D, H, centers, heights, query, sigma, rng);
      hills_eval_runtime(D, H, centers.data(), heights.data(), sigma, query.data(), &V_cpu,
                         grad_cpu.data());

      // --- upload / rebuild cost (separate; not folded into query timing) ---
      GpuHillsCtx ctx;
      auto t_up0 = std::chrono::steady_clock::now();
      if (!ctx.upload(D, H, centers, heights, sigma, query, kBatchQ)) {
        emit_json("gpu_upload_failed", D, H, 0, 1, 0.0, 0.0, 0, nullptr);
        return failures + 1;
      }
      if (!cuda_ok(cudaDeviceSynchronize(), "sync after upload")) return failures + 1;
      auto t_up1 = std::chrono::steady_clock::now();
      const double upload_us =
          std::chrono::duration<double, std::micro>(t_up1 - t_up0).count();
      emit_json("gpu_hills_upload", D, H, 0, 0, upload_us, 0.0, 1,
                "\"phase\":\"upload_rebuild_separate\"");

      // Prepare batch queries (first = scalar query; rest random near it).
      std::vector<double> host_queries(static_cast<size_t>(kBatchQ) * D);
      std::uniform_real_distribution<double> Uc(-0.5, 0.5);
      for (int q = 0; q < kBatchQ; ++q) {
        for (int d = 0; d < D; ++d) {
          host_queries[static_cast<size_t>(q) * D + d] =
              (q == 0) ? query[static_cast<size_t>(d)] : Uc(rng);
        }
      }
      if (!cuda_ok(cudaMemcpy(ctx.queries_batch.as<double>(), host_queries.data(),
                              ctx.queries_batch.bytes, cudaMemcpyHostToDevice),
                   "H2D batch queries"))
        return failures + 1;

      // CPU scalar baseline timing
      {
        const int iters = iters_for(D, H, 1, D >= 32);
        std::vector<double> samples;
        for (int w = 0; w < kWarmup; ++w) {
          hills_eval_runtime(D, H, centers.data(), heights.data(), sigma, query.data(), &V_cpu,
                             grad_cpu.data());
          consume(V_cpu);
        }
        for (int s = 0; s < kMedianSamples; ++s) {
          auto t0 = std::chrono::steady_clock::now();
          for (int i = 0; i < iters; ++i) {
          asm volatile("" ::: "memory"); // 阻止编译器把相同输入求值移出计时循环。
            hills_eval_runtime(D, H, centers.data(), heights.data(), sigma, query.data(), &V_cpu,
                               grad_cpu.data());
            consume(V_cpu);
            consume_vec(grad_cpu.data(), D);
          }
          auto t1 = std::chrono::steady_clock::now();
          samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                            static_cast<double>(iters));
        }
        emit_json("cpu_hills_scalar_ref", D, H, 0, 1, median_us(samples), 0.0, 1,
                  "\"role\":\"cpu_baseline\"");
      }

      for (int block : blocks) {
        const int nBlocks = (H + block - 1) / block;
        if (nBlocks > ctx.max_blocks) {
          // Realloc block scratch if needed (block=64 sets max; 64 gives most blocks).
          ctx.max_blocks = nBlocks;
          if (!cuda_ok(ctx.block_V.alloc(sizeof(double) * static_cast<size_t>(nBlocks)),
                       "realloc block_V") ||
              !cuda_ok(ctx.block_grad.alloc(sizeof(double) * static_cast<size_t>(nBlocks) * D),
                       "realloc block_grad"))
            return failures + 1;
        }

        // Accuracy vs CPU analytic
        if (!launch_single_query(D, H, block, ctx.centers.as<double>(), ctx.heights.as<double>(),
                                 sigma, ctx.query.as<double>(), ctx.block_V.as<double>(),
                                 ctx.block_grad.as<double>(), ctx.V.as<double>(),
                                 ctx.grad.as<double>(), stream.s)) {
          emit_json("gpu_hills_launch_failed", D, H, block, 1, 0.0, 0.0, 0, nullptr);
          ++failures;
          continue;
        }
        if (!cuda_ok(cudaStreamSynchronize(stream.s), "sync single")) return failures + 1;
        if (!cuda_ok(cudaMemcpy(&V_gpu, ctx.V.as<double>(), sizeof(double), cudaMemcpyDeviceToHost),
                     "D2H V"))
          return failures + 1;
        if (!cuda_ok(cudaMemcpy(grad_gpu.data(), ctx.grad.as<double>(),
                                sizeof(double) * static_cast<size_t>(D), cudaMemcpyDeviceToHost),
                     "D2H grad"))
          return failures + 1;
        const double err =
            std::max(std::abs(V_gpu - V_cpu), max_abs_diff(grad_gpu.data(), grad_cpu.data(), D));
        // Reduction order differs vs CPU; scale tol with magnitude and H.
        const double gpu_tol = std::max(1e-10, 1e-11 * std::max(1.0, std::abs(V_cpu)));
        const int pass = err <= gpu_tol;
        if (!pass) ++failures;
        {
          char extra[128];
          std::snprintf(extra, sizeof(extra), "\"nBlocks\":%d,\"ref\":\"cpu_analytic\"", nBlocks);
          emit_json("gpu_hills_accuracy", D, H, block, 1, 0.0, err, pass, extra);
        }

        // Resident GPU: hills already on device; H2D query + kernel + D2H V/grad + sync
        // (end-to-end includes transfer; reported as e2e). Also report resident-only
        // (query already on device) separately — still wall-clock chrono, not CUDA-event-only.
        const int iters = iters_for(D, H, 1, D >= 32 || H >= 10000);
        std::vector<double> samples_e2e, samples_resident;
        samples_e2e.reserve(kMedianSamples);
        samples_resident.reserve(kMedianSamples);

        for (int w = 0; w < kWarmup; ++w) {
          if (!cuda_ok(cudaMemcpyAsync(ctx.query.as<double>(), query.data(), ctx.query.bytes,
                                       cudaMemcpyHostToDevice, stream.s),
                       "warmup H2D"))
            return failures + 1;
          if (!launch_single_query(D, H, block, ctx.centers.as<double>(), ctx.heights.as<double>(),
                                   sigma, ctx.query.as<double>(), ctx.block_V.as<double>(),
                                   ctx.block_grad.as<double>(), ctx.V.as<double>(),
                                   ctx.grad.as<double>(), stream.s))
            return failures + 1;
          if (!cuda_ok(cudaMemcpyAsync(&V_gpu, ctx.V.as<double>(), sizeof(double),
                                       cudaMemcpyDeviceToHost, stream.s),
                       "warmup D2H V"))
            return failures + 1;
          if (!cuda_ok(cudaMemcpyAsync(grad_gpu.data(), ctx.grad.as<double>(),
                                       sizeof(double) * D, cudaMemcpyDeviceToHost, stream.s),
                       "warmup D2H g"))
            return failures + 1;
          if (!cuda_ok(cudaStreamSynchronize(stream.s), "warmup sync")) return failures + 1;
          consume(V_gpu);
          consume_vec(grad_gpu.data(), D);
        }

        for (int s = 0; s < kMedianSamples; ++s) {
          // End-to-end per query (includes transfers + stream sync)
          auto t0 = std::chrono::steady_clock::now();
          for (int i = 0; i < iters; ++i) {
          asm volatile("" ::: "memory"); // 阻止编译器把相同输入求值移出计时循环。
            if (cudaMemcpyAsync(ctx.query.as<double>(), query.data(), ctx.query.bytes,
                                cudaMemcpyHostToDevice, stream.s) != cudaSuccess)
              return failures + 1;
            if (!launch_single_query(D, H, block, ctx.centers.as<double>(),
                                     ctx.heights.as<double>(), sigma, ctx.query.as<double>(),
                                     ctx.block_V.as<double>(), ctx.block_grad.as<double>(),
                                     ctx.V.as<double>(), ctx.grad.as<double>(), stream.s))
              return failures + 1;
            if (cudaMemcpyAsync(&V_gpu, ctx.V.as<double>(), sizeof(double), cudaMemcpyDeviceToHost,
                                stream.s) != cudaSuccess)
              return failures + 1;
            if (cudaMemcpyAsync(grad_gpu.data(), ctx.grad.as<double>(), sizeof(double) * D,
                                cudaMemcpyDeviceToHost, stream.s) != cudaSuccess)
              return failures + 1;
            if (cudaStreamSynchronize(stream.s) != cudaSuccess) return failures + 1;
            consume(V_gpu);
            consume_vec(grad_gpu.data(), D);
          }
          auto t1 = std::chrono::steady_clock::now();
          samples_e2e.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                                static_cast<double>(iters));

          // Resident: hills+query already on device; wall = kernel + stream sync (no H2D/D2H).
          t0 = std::chrono::steady_clock::now();
          for (int i = 0; i < iters; ++i) {
          asm volatile("" ::: "memory"); // 阻止编译器把相同输入求值移出计时循环。
            if (!launch_single_query(D, H, block, ctx.centers.as<double>(),
                                     ctx.heights.as<double>(), sigma, ctx.query.as<double>(),
                                     ctx.block_V.as<double>(), ctx.block_grad.as<double>(),
                                     ctx.V.as<double>(), ctx.grad.as<double>(), stream.s))
              return failures + 1;
            if (cudaStreamSynchronize(stream.s) != cudaSuccess) return failures + 1;
          }
          t1 = std::chrono::steady_clock::now();
          if (cudaMemcpy(&V_gpu, ctx.V.as<double>(), sizeof(double), cudaMemcpyDeviceToHost) !=
              cudaSuccess)
            return failures + 1;
          if (cudaMemcpy(grad_gpu.data(), ctx.grad.as<double>(), sizeof(double) * D,
                         cudaMemcpyDeviceToHost) != cudaSuccess)
            return failures + 1;
          consume(V_gpu);
          consume_vec(grad_gpu.data(), D);
          samples_resident.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                                     static_cast<double>(iters));
        }

        emit_json("gpu_hills_e2e_per_query", D, H, block, 1, median_us(samples_e2e), err, pass,
                  "\"includes_transfer\":true,\"timing\":\"chrono_wall\"");
        emit_json("gpu_hills_resident", D, H, block, 1, median_us(samples_resident), err, pass,
                  "\"includes_transfer\":false,\"timing\":\"chrono_wall_kernel_sync\"");

        // Batched queries (e.g. 128): upload batch once outside timed region already done;
        // timed region = kernel + D2H results + sync (batch transfer of queries optional e2e).
        {
          const int Q = kBatchQ;
          // Skip huge D*H*Q if needed
          const int biters = iters_for(D, H, Q, true);
          std::vector<double> host_V(Q), host_G(static_cast<size_t>(Q) * D);
          std::vector<double> samples_batch, samples_batch_e2e;

          for (int w = 0; w < kWarmup; ++w) {
            hills_batched_kernel<<<Q, block, shmem_bytes(block, D), stream.s>>>(
                D, H, Q, ctx.centers.as<double>(), ctx.heights.as<double>(), sigma,
                ctx.queries_batch.as<double>(), ctx.V_batch.as<double>(),
                ctx.grad_batch.as<double>());
            if (cudaGetLastError() != cudaSuccess) return failures + 1;
            if (cudaStreamSynchronize(stream.s) != cudaSuccess) return failures + 1;
          }

          // Accuracy: compare query 0 to CPU
          hills_batched_kernel<<<Q, block, shmem_bytes(block, D), stream.s>>>(
              D, H, Q, ctx.centers.as<double>(), ctx.heights.as<double>(), sigma,
              ctx.queries_batch.as<double>(), ctx.V_batch.as<double>(), ctx.grad_batch.as<double>());
          if (!cuda_ok(cudaStreamSynchronize(stream.s), "batch sync")) return failures + 1;
          if (!cuda_ok(cudaMemcpy(host_V.data(), ctx.V_batch.as<double>(), sizeof(double) * Q,
                                  cudaMemcpyDeviceToHost),
                       "D2H batch V"))
            return failures + 1;
          if (!cuda_ok(cudaMemcpy(host_G.data(), ctx.grad_batch.as<double>(),
                                  sizeof(double) * static_cast<size_t>(Q) * D,
                                  cudaMemcpyDeviceToHost),
                       "D2H batch G"))
            return failures + 1;
          double berr=0;
          double reference_scale=1;
          for(int query_index=0;query_index<Q;++query_index) {
            double reference_energy;
            std::vector<double> reference_gradient(D);
            hills_eval_runtime(D,H,centers.data(),heights.data(),sigma,
                host_queries.data()+size_t(query_index)*D,&reference_energy,reference_gradient.data());
            berr=std::max(berr,std::abs(host_V[query_index]-reference_energy));
            berr=std::max(berr,max_abs_diff(host_G.data()+size_t(query_index)*D,reference_gradient.data(),D));
            reference_scale=std::max(reference_scale,std::abs(reference_energy));
            for(double derivative:reference_gradient) reference_scale=std::max(reference_scale,std::abs(derivative));
          }
          const int bpass=berr<=1e-11*reference_scale;
          if (!bpass) ++failures;

          for (int s = 0; s < kMedianSamples; ++s) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < biters; ++i) {
              hills_batched_kernel<<<Q, block, shmem_bytes(block, D), stream.s>>>(
                  D, H, Q, ctx.centers.as<double>(), ctx.heights.as<double>(), sigma,
                  ctx.queries_batch.as<double>(), ctx.V_batch.as<double>(),
                  ctx.grad_batch.as<double>());
              if (cudaStreamSynchronize(stream.s) != cudaSuccess) return failures + 1;
            }
            auto t1 = std::chrono::steady_clock::now();
            if (cudaMemcpy(host_V.data(), ctx.V_batch.as<double>(), sizeof(double) * Q,
                           cudaMemcpyDeviceToHost) != cudaSuccess)
              return failures + 1;
            if (cudaMemcpy(host_G.data(), ctx.grad_batch.as<double>(),
                           sizeof(double) * static_cast<size_t>(Q) * D,
                           cudaMemcpyDeviceToHost) != cudaSuccess)
              return failures + 1;
            consume(host_V[0]);
            consume_vec(host_G.data(), D);
            samples_batch.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                                    static_cast<double>(biters));

            // Batch e2e: also re-upload query batch each iter
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < biters; ++i) {
              if (cudaMemcpyAsync(ctx.queries_batch.as<double>(), host_queries.data(),
                                  ctx.queries_batch.bytes, cudaMemcpyHostToDevice, stream.s) !=
                  cudaSuccess)
                return failures + 1;
              hills_batched_kernel<<<Q, block, shmem_bytes(block, D), stream.s>>>(
                  D, H, Q, ctx.centers.as<double>(), ctx.heights.as<double>(), sigma,
                  ctx.queries_batch.as<double>(), ctx.V_batch.as<double>(),
                  ctx.grad_batch.as<double>());
              if (cudaMemcpyAsync(host_V.data(), ctx.V_batch.as<double>(), sizeof(double) * Q,
                                  cudaMemcpyDeviceToHost, stream.s) != cudaSuccess)
                return failures + 1;
              if (cudaMemcpyAsync(host_G.data(), ctx.grad_batch.as<double>(),
                                  sizeof(double) * static_cast<size_t>(Q) * D,
                                  cudaMemcpyDeviceToHost, stream.s) != cudaSuccess)
                return failures + 1;
              if (cudaStreamSynchronize(stream.s) != cudaSuccess) return failures + 1;
              consume(host_V[0]);
            }
            t1 = std::chrono::steady_clock::now();
            samples_batch_e2e.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count() /
                static_cast<double>(biters));
          }

          char extra[160];
          std::snprintf(extra, sizeof(extra),
                        "\"includes_transfer\":false,\"timing\":\"chrono_wall\",\"batch_err\":%.16e",
                        berr);
          emit_json("gpu_hills_batched_resident", D, H, block, Q, median_us(samples_batch), berr,
                    bpass, extra);
          std::snprintf(extra, sizeof(extra),
                        "\"includes_transfer\":true,\"timing\":\"chrono_wall\"");
          emit_json("gpu_hills_batched_e2e", D, H, block, Q, median_us(samples_batch_e2e), berr,
                    bpass, extra);
        }
      }
    }
  }
  return failures;
}

}  // namespace

int main() {
  const auto wall0 = std::chrono::steady_clock::now();
  int failures = 0;
  failures += run_cpu_hills();
  failures += run_cpu_grid();
  failures += run_gpu_hills();
  const auto wall1 = std::chrono::steady_clock::now();
  const double wall_s = std::chrono::duration<double>(wall1 - wall0).count();
  char extra[128];
  std::snprintf(extra, sizeof(extra), "\"wall_s\":%.3f,\"sink\":%.16e", wall_s, g_sink);
  emit_json("summary", 0, 0, 0, 0, wall_s * 1e6, 0.0, failures == 0, extra);
  if (failures != 0) {
    std::fprintf(stderr, "benchmark finished with %d failure(s)\n", failures);
    return 1;
  }
  return 0;
}
