#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#define POW2(a) ((a) * (a))
#define POW3(a) ((a) * (a) * (a))
#define POW4(a) ((a) * (a) * (a) * (a))
#define POW5(a) ((a) * (a) * (a) * (a) * (a))
#define POW6(a) (POW3(a) * POW3(a))
#define POW7(a) (POW3(a) * POW4(a))
#define PI 3.1415926535897932385

// This is the production project's generated/hand-optimized Ylm implementation.
#include "CV_Steinhardt_math.h"

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t e_ = (call);                                                     \
    if (e_ != cudaSuccess) {                                                     \
      std::cerr << "CUDA failure at " << __FILE__ << ':' << __LINE__ << ": "   \
                << cudaGetErrorString(e_) << std::endl;                          \
      std::exit(2);                                                              \
    }                                                                            \
  } while (0)

struct HostGraph {
  int n = 0;
  double box = 0.0;
  std::vector<int> offsets;
  std::vector<int> neighbors;
  std::vector<int> sources;
  std::vector<double4> dr;
  std::vector<int> incoming_offsets;
  std::vector<int> incoming_edges;
  std::vector<int> pair_i;
  std::vector<int> pair_j;
  std::vector<double4> pair_dr;
  std::vector<int> incidence_offsets;
  std::vector<int> signed_incidence;
};

struct DeviceGraph {
  int n = 0;
  int e = 0;
  int *offsets = nullptr;
  int *neighbors = nullptr;
  int *sources = nullptr;
  double4 *dr = nullptr;
  int *incoming_offsets = nullptr;
  int *incoming_edges = nullptr;
  int npairs = 0;
  int *pair_i = nullptr;
  int *pair_j = nullptr;
  double4 *pair_dr = nullptr;
  int *incidence_offsets = nullptr;
  int *signed_incidence = nullptr;

  explicit DeviceGraph(const HostGraph &h) : n(h.n), e(static_cast<int>(h.neighbors.size())),
                                              npairs(static_cast<int>(h.pair_i.size())) {
    CUDA_CHECK(cudaMalloc(&offsets, sizeof(int) * (n + 1)));
    CUDA_CHECK(cudaMalloc(&neighbors, sizeof(int) * e));
    CUDA_CHECK(cudaMalloc(&sources, sizeof(int) * e));
    CUDA_CHECK(cudaMalloc(&dr, sizeof(double4) * e));
    CUDA_CHECK(cudaMalloc(&incoming_offsets, sizeof(int) * (n + 1)));
    CUDA_CHECK(cudaMalloc(&incoming_edges, sizeof(int) * e));
    CUDA_CHECK(cudaMalloc(&pair_i, sizeof(int) * npairs));
    CUDA_CHECK(cudaMalloc(&pair_j, sizeof(int) * npairs));
    CUDA_CHECK(cudaMalloc(&pair_dr, sizeof(double4) * npairs));
    CUDA_CHECK(cudaMalloc(&incidence_offsets, sizeof(int) * (n + 1)));
    CUDA_CHECK(cudaMalloc(&signed_incidence, sizeof(int) * (2 * npairs)));
    CUDA_CHECK(cudaMemcpy(offsets, h.offsets.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(neighbors, h.neighbors.data(), sizeof(int) * e, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sources, h.sources.data(), sizeof(int) * e, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dr, h.dr.data(), sizeof(double4) * e, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(incoming_offsets, h.incoming_offsets.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(incoming_edges, h.incoming_edges.data(), sizeof(int) * e, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pair_i, h.pair_i.data(), sizeof(int) * npairs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pair_j, h.pair_j.data(), sizeof(int) * npairs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pair_dr, h.pair_dr.data(), sizeof(double4) * npairs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(incidence_offsets, h.incidence_offsets.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(signed_incidence, h.signed_incidence.data(), sizeof(int) * (2*npairs), cudaMemcpyHostToDevice));
  }
  ~DeviceGraph() {
    cudaFree(offsets);
    cudaFree(neighbors);
    cudaFree(sources);
    cudaFree(dr);
    cudaFree(incoming_offsets);
    cudaFree(incoming_edges);
    cudaFree(pair_i);
    cudaFree(pair_j);
    cudaFree(pair_dr);
    cudaFree(incidence_offsets);
    cudaFree(signed_incidence);
  }
};

static double minimum_image(double x, double box) {
  return x - box * std::nearbyint(x / box);
}

static HostGraph make_fcc_graph(int nc, double cutoff) {
  constexpr double jitter = 0.006;
  const std::array<std::array<double, 3>, 4> basis{{
      {{0.0, 0.0, 0.0}}, {{0.0, 0.5, 0.5}},
      {{0.5, 0.0, 0.5}}, {{0.5, 0.5, 0.0}}}};
  const int n = 4 * nc * nc * nc;
  const double box = static_cast<double>(nc);
  std::vector<std::array<double, 3>> x(n);
  auto id = [nc](int ix, int iy, int iz, int b) {
    return (((iz * nc + iy) * nc + ix) * 4 + b);
  };
  auto wrap_cell = [nc](int v) {
    v %= nc;
    return v < 0 ? v + nc : v;
  };
  for (int iz = 0; iz < nc; ++iz) {
    for (int iy = 0; iy < nc; ++iy) {
      for (int ix = 0; ix < nc; ++ix) {
        for (int b = 0; b < 4; ++b) {
          int a = id(ix, iy, iz, b);
          for (int d = 0; d < 3; ++d) {
            double p = (d == 0 ? ix : (d == 1 ? iy : iz)) + basis[b][d];
            p += jitter * std::sin(0.754877666 * (a + 1) * (d + 2));
            p -= box * std::floor(p / box);
            x[a][d] = p;
          }
        }
      }
    }
  }

  HostGraph g;
  g.n = n;
  g.box = box;
  g.offsets.resize(n + 1, 0);
  const int span = static_cast<int>(std::ceil(cutoff)) + 1;
  const double cut2 = cutoff * cutoff;
  for (int iz = 0; iz < nc; ++iz) {
    for (int iy = 0; iy < nc; ++iy) {
      for (int ix = 0; ix < nc; ++ix) {
        for (int b = 0; b < 4; ++b) {
          const int i = id(ix, iy, iz, b);
          std::vector<std::pair<int, double4>> row;
          for (int dz = -span; dz <= span; ++dz) {
            for (int dy = -span; dy <= span; ++dy) {
              for (int dx = -span; dx <= span; ++dx) {
                int jx = wrap_cell(ix + dx);
                int jy = wrap_cell(iy + dy);
                int jz = wrap_cell(iz + dz);
                for (int bj = 0; bj < 4; ++bj) {
                  int j = id(jx, jy, jz, bj);
                  if (j == i) continue;
                  double rx = minimum_image(x[j][0] - x[i][0], box);
                  double ry = minimum_image(x[j][1] - x[i][1], box);
                  double rz = minimum_image(x[j][2] - x[i][2], box);
                  double r2 = rx * rx + ry * ry + rz * rz;
                  if (r2 < cut2 && r2 > 1e-20) row.push_back({j, make_double4(rx, ry, rz, r2)});
                }
              }
            }
          }
          std::sort(row.begin(), row.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
          row.erase(std::unique(row.begin(), row.end(), [](const auto &a, const auto &b) {
                      return a.first == b.first;
                    }), row.end());
          for (const auto &p : row) {
            g.sources.push_back(i);
            g.neighbors.push_back(p.first);
            g.dr.push_back(p.second);
          }
          g.offsets[i + 1] = static_cast<int>(g.neighbors.size());
        }
      }
    }
  }

  g.incoming_offsets.assign(n + 1, 0);
  for (int j : g.neighbors) ++g.incoming_offsets[j + 1];
  std::partial_sum(g.incoming_offsets.begin(), g.incoming_offsets.end(), g.incoming_offsets.begin());
  g.incoming_edges.resize(g.neighbors.size());
  std::vector<int> cursor = g.incoming_offsets;
  for (int e = 0; e < static_cast<int>(g.neighbors.size()); ++e) {
    g.incoming_edges[cursor[g.neighbors[e]]++] = e;
  }

  for (int e = 0; e < static_cast<int>(g.neighbors.size()); ++e) {
    int i = g.sources[e], j = g.neighbors[e];
    if (i < j) {
      g.pair_i.push_back(i);
      g.pair_j.push_back(j);
      g.pair_dr.push_back(g.dr[e]);
    }
  }
  g.incidence_offsets.assign(n + 1, 0);
  for (size_t p = 0; p < g.pair_i.size(); ++p) {
    ++g.incidence_offsets[g.pair_i[p] + 1];
    ++g.incidence_offsets[g.pair_j[p] + 1];
  }
  std::partial_sum(g.incidence_offsets.begin(), g.incidence_offsets.end(), g.incidence_offsets.begin());
  g.signed_incidence.resize(2 * g.pair_i.size());
  cursor = g.incidence_offsets;
  for (int p = 0; p < static_cast<int>(g.pair_i.size()); ++p) {
    g.signed_incidence[cursor[g.pair_i[p]]++] = p + 1;
    g.signed_incidence[cursor[g.pair_j[p]]++] = -(p + 1);
  }
  return g;
}

template <int L>
struct Trig {
  double r;
  double ct, st, cp, sp;
  double c2t, s2t, c2p, s2p;
  double c3t, s3t, c3p, s3p;
  double c4t, s4t, c4p, s4p;
  double c5t, s5t, c5p, s5p;
  double c6t, s6t, c6p, s6p;
};

template <int L>
__device__ __forceinline__ Trig<L> make_trig(const double4 v) {
  Trig<L> t{};
  t.r = sqrt(v.w);
  double theta = acos(fmax(-1.0, fmin(1.0, v.z / t.r)));
  double phi = atan2(v.y, v.x);
  sincos(theta, &t.st, &t.ct);
  sincos(phi, &t.sp, &t.cp);
  if constexpr (L >= 2) {
    sincos(2.0 * theta, &t.s2t, &t.c2t);
    sincos(2.0 * phi, &t.s2p, &t.c2p);
  }
  if constexpr (L >= 3) {
    sincos(3.0 * phi, &t.s3p, &t.c3p);
    sincos(3.0 * theta, &t.s3t, &t.c3t);
    sincos(4.0 * theta, &t.s4t, &t.c4t);
  }
  if constexpr (L >= 4) {
    sincos(4.0 * phi, &t.s4p, &t.c4p);
    sincos(5.0 * theta, &t.s5t, &t.c5t);
    sincos(5.0 * phi, &t.s5p, &t.c5p);
    sincos(6.0 * theta, &t.s6t, &t.c6t);
  }
  if constexpr (L >= 6) sincos(6.0 * phi, &t.s6p, &t.c6p);
  return t;
}

template <int L>
__device__ __forceinline__ void forward_one(const Trig<L> &t, double *q, double *y) {
  if constexpr (L == 4) {
    compute_qlm_forward_L4(1.0, t.ct, t.st, t.cp, t.sp,
                           t.c2p, t.s2p, t.c3p, t.s3p, t.c4p, t.s4p, q, y);
  } else {
    compute_qlm_forward_L6(1.0, t.ct, t.st, t.cp, t.sp,
                           t.c2t, t.s2t, t.c2p, t.s2p, t.c3p, t.s3p,
                           t.c4t, t.s4t, t.c4p, t.s4p, t.c5p, t.s5p,
                           t.c6t, t.s6t, t.c6p, t.s6p, q, y);
  }
}

template <int L>
__device__ __forceinline__ void gradient_one(const Trig<L> &t,
                                             double ci, double cj,
                                             int bi, int bj,
                                             double *qlm, double out[6]) {
  if constexpr (L == 4) {
    compute_Ylm_gradient_L4(t.r, t.ct, t.st, t.cp, t.sp,
        t.c2t, t.s2t, t.c2p, t.s2p, t.c3t, t.s3t, t.c3p, t.s3p,
        t.c4t, t.s4t, t.c4p, t.s4p, t.c5t, t.s5t, t.c5p, t.s5p,
        t.c6t, t.s6t, ci, cj, bi, bj, qlm, out);
  } else {
    compute_Ylm_gradient_L6(t.r, t.ct, t.st, t.cp, t.sp,
        t.c2t, t.s2t, t.c2p, t.s2p, t.c3t, t.s3t, t.c3p, t.s3p,
        t.c4t, t.s4t, t.c4p, t.s4p, t.c5t, t.s5t, t.c5p, t.s5p,
        t.c6t, t.s6t, t.c6p, t.s6p, ci, cj, bi, bj, qlm, out);
  }
}

template <int L>
__global__ void q_center_kernel(int n, const int *offsets, const double4 *dr,
                                double *qlm, double *ylm, double *ql) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  constexpr int lm = 2 * (L + 1);
  double qloc[lm] = {0.0};
  int begin = offsets[i], end = offsets[i + 1];
  if (begin == end) return;
  for (int e = begin; e < end; ++e) {
    Trig<L> t = make_trig<L>(dr[e]);
    forward_one<L>(t, qloc, ylm + static_cast<size_t>(e) * lm);
  }
  double inv = 1.0 / (end - begin);
  #pragma unroll
  for (int m = 0; m < lm; ++m) {
    qloc[m] *= inv;
    qlm[static_cast<size_t>(i) * lm + m] = qloc[m];
  }
  double s = qloc[0] * qloc[0];
  #pragma unroll
  for (int m = 1; m <= L; ++m) s += 2.0 * (POW2(qloc[2*m]) + POW2(qloc[2*m+1]));
  ql[i] = sqrt(s * (4.0 * PI) / (2 * L + 1));
}

template <int L>
__global__ void q_edge_kernel(int ecount, const double4 *dr, double *ylm) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= ecount) return;
  constexpr int lm = 2 * (L + 1);
  double dummy[lm] = {0.0};
  Trig<L> t = make_trig<L>(dr[e]);
  forward_one<L>(t, dummy, ylm + static_cast<size_t>(e) * lm);
}

// Mirrors LocalQ's forward decomposition: the first kernel performs the same
// center-owned neighbor loop and stores qlm/Ylm, while a second kernel derives
// the scalar q value from the stored qlm.
template <int L>
__global__ void q_center_qlm_kernel(int n, const int *offsets, const double4 *dr,
                                    double *qlm, double *ylm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  constexpr int lm = 2 * (L + 1);
  double qloc[lm] = {0.0};
  int begin = offsets[i], end = offsets[i + 1];
  if (begin == end) return;
  for (int e = begin; e < end; ++e) {
    Trig<L> t = make_trig<L>(dr[e]);
    forward_one<L>(t, qloc, ylm + static_cast<size_t>(e) * lm);
  }
  double inv = 1.0 / (end - begin);
  #pragma unroll
  for (int m = 0; m < lm; ++m) qlm[static_cast<size_t>(i) * lm + m] = qloc[m] * inv;
}

template <int L>
__global__ void q_finalize_kernel(int n, const double *qlm, double *ql) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  constexpr int lm = 2 * (L + 1);
  const double *q = qlm + static_cast<size_t>(i) * lm;
  double s = q[0] * q[0];
  #pragma unroll
  for (int m = 1; m <= L; ++m) s += 2.0 * (POW2(q[2*m]) + POW2(q[2*m+1]));
  ql[i] = sqrt(s * (4.0 * PI) / (2 * L + 1));
}

template <int L>
__global__ void q_reduce_kernel(int n, const int *offsets, const double *ylm,
                                double *qlm, double *ql) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  constexpr int lm = 2 * (L + 1);
  double qloc[lm] = {0.0};
  int begin = offsets[i], end = offsets[i + 1];
  if (begin == end) return;
  for (int e = begin; e < end; ++e) {
    #pragma unroll
    for (int m = 0; m < lm; ++m) qloc[m] += ylm[static_cast<size_t>(e) * lm + m];
  }
  double inv = 1.0 / (end - begin);
  #pragma unroll
  for (int m = 0; m < lm; ++m) {
    qloc[m] *= inv;
    qlm[static_cast<size_t>(i) * lm + m] = qloc[m];
  }
  double s = qloc[0] * qloc[0];
  #pragma unroll
  for (int m = 1; m <= L; ++m) s += 2.0 * (POW2(qloc[2*m]) + POW2(qloc[2*m+1]));
  ql[i] = sqrt(s * (4.0 * PI) / (2 * L + 1));
}

template <int L>
__global__ void force_original_gather(int n, const int *offsets, const int *neighbors,
                                      const double4 *dr, double *qlm, const double *ql,
                                      double *grad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  constexpr int lm = 2 * (L + 1);
  double sum[6] = {0.0};
  int begin = offsets[i], end = offsets[i + 1];
  if (begin == end) { grad[3*i]=grad[3*i+1]=grad[3*i+2]=0.0; return; }
  double ci = 1.0 / (ql[i] * (end - begin));
  for (int e = begin; e < end; ++e) {
    int j = neighbors[e];
    int nbj = offsets[j + 1] - offsets[j];
    double cj = 1.0 / (ql[j] * nbj);
    double tmp[6] = {0.0};
    gradient_one<L>(make_trig<L>(dr[e]), ci, cj, i * lm, j * lm, qlm, tmp);
    #pragma unroll
    for (int k = 0; k < 6; ++k) sum[k] += tmp[k];
  }
  double scale = -(2.0 * PI) / (n * (2 * L + 1));
  grad[3*i+0] = scale * (sum[0] + sum[1]);
  grad[3*i+1] = scale * (sum[2] + sum[3]);
  grad[3*i+2] = scale * (sum[4] + sum[5]);
}

template <int L>
__device__ __forceinline__ double3 source_center_term(int n, int i, const double4 v,
                                                       int degree, double *qlm,
                                                       const double *ql) {
  constexpr int lm = 2 * (L + 1);
  double tmp[6] = {0.0};
  double ci = 1.0 / (ql[i] * degree);
  gradient_one<L>(make_trig<L>(v), ci, 0.0, i * lm, 0, qlm, tmp);
  double scale = -(2.0 * PI) / (n * (2 * L + 1));
  return make_double3(scale * (tmp[0] + tmp[1]),
                      scale * (tmp[2] + tmp[3]),
                      scale * (tmp[4] + tmp[5]));
}

// Structurally mirrors LocalQ: one thread per source center, atomics to each
// neighbor, then atomics for the accumulated center contribution.
template <int L>
__global__ void force_localq_atomic(int n, const int *offsets, const int *neighbors,
                                    const double4 *dr, double *qlm, const double *ql,
                                    double *grad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int begin = offsets[i], end = offsets[i + 1], degree = end - begin;
  double3 self = make_double3(0.0, 0.0, 0.0);
  for (int e = begin; e < end; ++e) {
    double3 c = source_center_term<L>(n, i, dr[e], degree, qlm, ql);
    self.x += c.x; self.y += c.y; self.z += c.z;
    int j = neighbors[e];
    atomicAdd(grad + 3*j+0, -c.x);
    atomicAdd(grad + 3*j+1, -c.y);
    atomicAdd(grad + 3*j+2, -c.z);
  }
  atomicAdd(grad + 3*i+0, self.x);
  atomicAdd(grad + 3*i+1, self.y);
  atomicAdd(grad + 3*i+2, self.z);
}

// Same center-oriented scatter, but the center owns a conflict-free output.
// Only neighbor contributions use atomics.
template <int L>
__global__ void force_center_split(int n, const int *offsets, const int *neighbors,
                                   const double4 *dr, double *qlm, const double *ql,
                                   double *self_grad, double *neighbor_grad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int begin = offsets[i], end = offsets[i + 1], degree = end - begin;
  double3 self = make_double3(0.0, 0.0, 0.0);
  for (int e = begin; e < end; ++e) {
    double3 c = source_center_term<L>(n, i, dr[e], degree, qlm, ql);
    self.x += c.x; self.y += c.y; self.z += c.z;
    int j = neighbors[e];
    atomicAdd(neighbor_grad + 3*j+0, -c.x);
    atomicAdd(neighbor_grad + 3*j+1, -c.y);
    atomicAdd(neighbor_grad + 3*j+2, -c.z);
  }
  self_grad[3*i+0] = self.x;
  self_grad[3*i+1] = self.y;
  self_grad[3*i+2] = self.z;
}

template <int L>
__global__ void force_warp_center_split(int n, const int *offsets, const int *neighbors,
                                        const double4 *dr, double *qlm, const double *ql,
                                        double *self_grad, double *neighbor_grad) {
  int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp_global >= n) return;
  int i = warp_global;
  int begin = offsets[i], end = offsets[i + 1], degree = end - begin;
  double3 self = make_double3(0.0, 0.0, 0.0);
  for (int e = begin + lane; e < end; e += 32) {
    double3 c = source_center_term<L>(n, i, dr[e], degree, qlm, ql);
    self.x += c.x; self.y += c.y; self.z += c.z;
    int j = neighbors[e];
    atomicAdd(neighbor_grad + 3*j+0, -c.x);
    atomicAdd(neighbor_grad + 3*j+1, -c.y);
    atomicAdd(neighbor_grad + 3*j+2, -c.z);
  }
  // The neighbor loop diverges when degree is not a multiple of 32.  Force a
  // full-warp reconvergence before the reduction; using __activemask() here
  // can capture only the lanes from the last loop trip on independently
  // scheduled GPUs.
  __syncwarp(0xffffffffu);
  constexpr unsigned mask = 0xffffffffu;
  for (int delta = 16; delta > 0; delta >>= 1) {
    self.x += __shfl_down_sync(mask, self.x, delta);
    self.y += __shfl_down_sync(mask, self.y, delta);
    self.z += __shfl_down_sync(mask, self.z, delta);
  }
  if (lane == 0) {
    self_grad[3*i+0] = self.x;
    self_grad[3*i+1] = self.y;
    self_grad[3*i+2] = self.z;
  }
}

__global__ void combine_grad(int n, const double *a, const double *b, double *out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < 3*n) out[k] = a[k] + b[k];
}

__global__ void perturb_edges(int ecount, const int *sources, const int *neighbors,
                              const double4 *base, int atom, int dim,
                              double delta, double4 *out) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= ecount) return;
  double4 v = base[e];
  double shift = (neighbors[e] == atom ? delta : 0.0)
               - (sources[e] == atom ? delta : 0.0);
  if (dim == 0) v.x += shift;
  if (dim == 1) v.y += shift;
  if (dim == 2) v.z += shift;
  v.w = v.x*v.x + v.y*v.y + v.z*v.z;
  out[e] = v;
}

// Maximum edge parallelism, with atomics at both endpoints.
template <int L>
__global__ void force_edge_atomic(int n, int ecount, const int *sources,
                                  const int *neighbors, const int *offsets,
                                  const double4 *dr, double *qlm, const double *ql,
                                  double *grad) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= ecount) return;
  int i = sources[e], j = neighbors[e];
  int degree = offsets[i + 1] - offsets[i];
  double3 c = source_center_term<L>(n, i, dr[e], degree, qlm, ql);
  atomicAdd(grad + 3*i+0, c.x); atomicAdd(grad + 3*i+1, c.y); atomicAdd(grad + 3*i+2, c.z);
  atomicAdd(grad + 3*j+0, -c.x); atomicAdd(grad + 3*j+1, -c.y); atomicAdd(grad + 3*j+2, -c.z);
}

template <int L>
__global__ void force_edge_store(int n, int ecount, const int *sources,
                                 const int *offsets, const double4 *dr,
                                 double *qlm, const double *ql, double3 *edge_value) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= ecount) return;
  int i = sources[e];
  edge_value[e] = source_center_term<L>(n, i, dr[e], offsets[i+1]-offsets[i], qlm, ql);
}

__global__ void force_reverse_gather(int n, const int *offsets,
                                     const int *incoming_offsets,
                                     const int *incoming_edges,
                                     const double3 *edge_value, double *grad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double3 v = make_double3(0.0, 0.0, 0.0);
  for (int e = offsets[i]; e < offsets[i+1]; ++e) {
    v.x += edge_value[e].x; v.y += edge_value[e].y; v.z += edge_value[e].z;
  }
  for (int p = incoming_offsets[i]; p < incoming_offsets[i+1]; ++p) {
    int e = incoming_edges[p];
    v.x -= edge_value[e].x; v.y -= edge_value[e].y; v.z -= edge_value[e].z;
  }
  grad[3*i+0] = v.x; grad[3*i+1] = v.y; grad[3*i+2] = v.z;
}

// One full helper call per undirected pair.  The production helper already
// combines the q_i and q_j chains, so the opposite endpoint receives -c.
template <int L>
__global__ void force_pair_fused_atomic(int n, int npairs, const int *pair_i,
                                        const int *pair_j, const double4 *pair_dr,
                                        const int *offsets, double *qlm,
                                        const double *ql, double *grad) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npairs) return;
  constexpr int lm = 2 * (L + 1);
  int i = pair_i[p], j = pair_j[p];
  double ci = 1.0 / (ql[i] * (offsets[i+1]-offsets[i]));
  double cj = 1.0 / (ql[j] * (offsets[j+1]-offsets[j]));
  double tmp[6] = {0.0};
  gradient_one<L>(make_trig<L>(pair_dr[p]),ci,cj,i*lm,j*lm,qlm,tmp);
  double scale = -(2.0 * PI) / (n * (2 * L + 1));
  double3 c = make_double3(scale*(tmp[0]+tmp[1]),
                           scale*(tmp[2]+tmp[3]),
                           scale*(tmp[4]+tmp[5]));
  atomicAdd(grad+3*i+0,c.x); atomicAdd(grad+3*i+1,c.y); atomicAdd(grad+3*i+2,c.z);
  atomicAdd(grad+3*j+0,-c.x); atomicAdd(grad+3*j+1,-c.y); atomicAdd(grad+3*j+2,-c.z);
}

template <int L>
__global__ void force_pair_fused_store(int n, int npairs, const int *pair_i,
                                       const int *pair_j, const double4 *pair_dr,
                                       const int *offsets, double *qlm,
                                       const double *ql, double3 *pair_value) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npairs) return;
  constexpr int lm = 2 * (L + 1);
  int i = pair_i[p], j = pair_j[p];
  double ci = 1.0 / (ql[i] * (offsets[i+1]-offsets[i]));
  double cj = 1.0 / (ql[j] * (offsets[j+1]-offsets[j]));
  double tmp[6] = {0.0};
  gradient_one<L>(make_trig<L>(pair_dr[p]),ci,cj,i*lm,j*lm,qlm,tmp);
  double scale = -(2.0 * PI) / (n * (2 * L + 1));
  pair_value[p] = make_double3(scale*(tmp[0]+tmp[1]),
                               scale*(tmp[2]+tmp[3]),
                               scale*(tmp[4]+tmp[5]));
}

__global__ void force_incidence_gather(int n, const int *incidence_offsets,
                                       const int *signed_incidence,
                                       const double3 *pair_value, double *grad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double3 v = make_double3(0.0,0.0,0.0);
  for (int k=incidence_offsets[i]; k<incidence_offsets[i+1]; ++k) {
    int code=signed_incidence[k];
    int p=abs(code)-1;
    double sign=code>0 ? 1.0 : -1.0;
    v.x += sign*pair_value[p].x;
    v.y += sign*pair_value[p].y;
    v.z += sign*pair_value[p].z;
  }
  grad[3*i+0]=v.x; grad[3*i+1]=v.y; grad[3*i+2]=v.z;
}

static double median_time_ms(const std::function<void()> &launch, int repetitions,
                             int batches = 7) {
  for (int i = 0; i < 5; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> samples;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int b = 0; b < batches; ++b) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repetitions; ++r) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(ms / repetitions);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(samples.begin(), samples.end());
  return samples[samples.size()/2];
}

static std::vector<double> copy_vector(double *p, size_t n) {
  std::vector<double> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), p, sizeof(double)*n, cudaMemcpyDeviceToHost));
  return h;
}

static std::tuple<double,double,double> compare(const std::vector<double> &a,
                                                const std::vector<double> &b) {
  double max_abs = 0.0, ref_max = 0.0, rms = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::abs(a[i]-b[i]);
    max_abs = std::max(max_abs, d);
    ref_max = std::max(ref_max, std::abs(a[i]));
    rms += d*d;
  }
  rms = std::sqrt(rms/a.size());
  return {max_abs, max_abs/std::max(ref_max, 1e-300), rms};
}

template <int L>
static void run_case(const HostGraph &hg, int reps) {
  DeviceGraph g(hg);
  constexpr int lm = 2 * (L + 1);
  const int block = 128;
  const int atom_grid = (g.n + block - 1) / block;
  const int value_grid = (3*g.n + block - 1) / block;
  const int edge_grid = (g.e + block - 1) / block;
  const int pair_grid = (g.npairs + block - 1) / block;
  const int warps_per_block = block/32;
  const int warp_grid = (g.n + warps_per_block - 1) / warps_per_block;
  double *qlm, *qlm2, *ylm, *ql, *ql2;
  double *grad_ref, *grad, *self_grad, *neighbor_grad;
  double3 *edge_value;
  double3 *pair_value;
  double4 *dr_work;
  CUDA_CHECK(cudaMalloc(&qlm, sizeof(double)*g.n*lm));
  CUDA_CHECK(cudaMalloc(&qlm2, sizeof(double)*g.n*lm));
  CUDA_CHECK(cudaMalloc(&ylm, sizeof(double)*static_cast<size_t>(g.e)*lm));
  CUDA_CHECK(cudaMalloc(&ql, sizeof(double)*g.n));
  CUDA_CHECK(cudaMalloc(&ql2, sizeof(double)*g.n));
  CUDA_CHECK(cudaMalloc(&grad_ref, sizeof(double)*3*g.n));
  CUDA_CHECK(cudaMalloc(&grad, sizeof(double)*3*g.n));
  CUDA_CHECK(cudaMalloc(&self_grad, sizeof(double)*3*g.n));
  CUDA_CHECK(cudaMalloc(&neighbor_grad, sizeof(double)*3*g.n));
  CUDA_CHECK(cudaMalloc(&edge_value, sizeof(double3)*g.e));
  CUDA_CHECK(cudaMalloc(&pair_value, sizeof(double3)*g.npairs));
  CUDA_CHECK(cudaMalloc(&dr_work, sizeof(double4)*g.e));

  auto q_center = [&] {
    q_center_kernel<L><<<atom_grid,block>>>(g.n,g.offsets,g.dr,qlm,ylm,ql);
  };
  auto q_split = [&] {
    q_edge_kernel<L><<<edge_grid,block>>>(g.e,g.dr,ylm);
    q_reduce_kernel<L><<<atom_grid,block>>>(g.n,g.offsets,ylm,qlm2,ql2);
  };
  auto q_localq_style = [&] {
    q_center_qlm_kernel<L><<<atom_grid,block>>>(g.n,g.offsets,g.dr,qlm2,ylm);
    q_finalize_kernel<L><<<atom_grid,block>>>(g.n,qlm2,ql2);
  };
  q_center();
  q_localq_style();
  CUDA_CHECK(cudaDeviceSynchronize());
  auto hq = copy_vector(ql, g.n);
  auto hq2 = copy_vector(ql2, g.n);
  auto qerr_localq = compare(hq,hq2);
  q_split();
  CUDA_CHECK(cudaDeviceSynchronize());
  hq2 = copy_vector(ql2, g.n);
  auto qerr = compare(hq,hq2);

  // Keep the GPU in a stable P-state before sub-millisecond timings.  A long
  // first warmup is enough because all cases are run back-to-back.
  static bool globally_warmed = false;
  int warm_launches = globally_warmed ? 50 : 3000;
  for (int w = 0; w < warm_launches; ++w) q_center();
  CUDA_CHECK(cudaDeviceSynchronize());
  globally_warmed = true;

  auto original = [&] {
    force_original_gather<L><<<atom_grid,block>>>(g.n,g.offsets,g.neighbors,g.dr,qlm,ql,grad_ref);
  };
  auto localq = [&] {
    CUDA_CHECK(cudaMemsetAsync(grad,0,sizeof(double)*3*g.n));
    force_localq_atomic<L><<<atom_grid,block>>>(g.n,g.offsets,g.neighbors,g.dr,qlm,ql,grad);
  };
  auto center_split = [&] {
    CUDA_CHECK(cudaMemsetAsync(neighbor_grad,0,sizeof(double)*3*g.n));
    force_center_split<L><<<atom_grid,block>>>(g.n,g.offsets,g.neighbors,g.dr,qlm,ql,self_grad,neighbor_grad);
    combine_grad<<<value_grid,block>>>(g.n,self_grad,neighbor_grad,grad);
  };
  auto warp_split = [&] {
    CUDA_CHECK(cudaMemsetAsync(neighbor_grad,0,sizeof(double)*3*g.n));
    force_warp_center_split<L><<<warp_grid,block>>>(g.n,g.offsets,g.neighbors,g.dr,qlm,ql,self_grad,neighbor_grad);
    combine_grad<<<value_grid,block>>>(g.n,self_grad,neighbor_grad,grad);
  };
  auto edge_atomic = [&] {
    CUDA_CHECK(cudaMemsetAsync(grad,0,sizeof(double)*3*g.n));
    force_edge_atomic<L><<<edge_grid,block>>>(g.n,g.e,g.sources,g.neighbors,g.offsets,g.dr,qlm,ql,grad);
  };
  auto edge_gather = [&] {
    force_edge_store<L><<<edge_grid,block>>>(g.n,g.e,g.sources,g.offsets,g.dr,qlm,ql,edge_value);
    force_reverse_gather<<<atom_grid,block>>>(g.n,g.offsets,g.incoming_offsets,g.incoming_edges,edge_value,grad);
  };
  auto pair_atomic = [&] {
    CUDA_CHECK(cudaMemsetAsync(grad,0,sizeof(double)*3*g.n));
    force_pair_fused_atomic<L><<<pair_grid,block>>>(g.n,g.npairs,g.pair_i,g.pair_j,g.pair_dr,g.offsets,qlm,ql,grad);
  };
  auto pair_gather = [&] {
    force_pair_fused_store<L><<<pair_grid,block>>>(g.n,g.npairs,g.pair_i,g.pair_j,g.pair_dr,g.offsets,qlm,ql,pair_value);
    force_incidence_gather<<<atom_grid,block>>>(g.n,g.incidence_offsets,g.signed_incidence,pair_value,grad);
  };

  original(); CUDA_CHECK(cudaDeviceSynchronize());
  auto href = copy_vector(grad_ref, 3*g.n);

  // Independent central finite-difference oracle for d[(sum_i q_i)/N]/dx.
  // The topology is held fixed; epsilon is far below either cutoff margin.
  const double eps = 1e-5;
  double fd_max_abs = 0.0, fd_ref_max = 0.0;
  for (int atom : std::array<int,3>{{0, g.n/3, 2*g.n/3}}) {
    for (int dim = 0; dim < 3; ++dim) {
      double cv[2]{};
      for (int sign_idx = 0; sign_idx < 2; ++sign_idx) {
        double delta = sign_idx == 0 ? -eps : eps;
        perturb_edges<<<edge_grid,block>>>(g.e,g.sources,g.neighbors,g.dr,atom,dim,delta,dr_work);
        q_center_kernel<L><<<atom_grid,block>>>(g.n,g.offsets,dr_work,qlm2,ylm,ql2);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto vals = copy_vector(ql2,g.n);
        cv[sign_idx] = std::accumulate(vals.begin(),vals.end(),0.0)/g.n;
      }
      double fd = (cv[1]-cv[0])/(2.0*eps);
      double analytic = href[3*atom+dim];
      fd_max_abs = std::max(fd_max_abs,std::abs(fd-analytic));
      fd_ref_max = std::max(fd_ref_max,std::abs(analytic));
    }
  }
  struct Method { const char *name; std::function<void()> launch; const char *atomics; };
  std::vector<Method> methods{
      {"original_gather", original, "0"},
      {"localq_atomic", localq, "3E+3N"},
      {"center_split", center_split, "3E"},
      {"warp_center_split", warp_split, "3E"},
      {"edge_atomic", edge_atomic, "6E"},
      {"edge_reverse_gather", edge_gather, "0"},
      {"pair_fused_atomic", pair_atomic, "3E"},
      {"pair_fused_gather", pair_gather, "0"}};

  std::cout << "CASE,L=" << L << ",N=" << g.n << ",E=" << g.e
            << ",avg_neighbors=" << std::fixed << std::setprecision(2)
            << static_cast<double>(g.e)/g.n << '\n';
  std::cout << "QCHECK,localq_style,max_abs=" << std::scientific << std::get<0>(qerr_localq)
            << ",max_rel=" << std::get<1>(qerr_localq) << ",rms=" << std::get<2>(qerr_localq) << '\n';
  std::cout << "QCHECK,edge_parallel,max_abs=" << std::scientific << std::get<0>(qerr)
            << ",max_rel=" << std::get<1>(qerr) << ",rms=" << std::get<2>(qerr) << '\n';
  std::cout << "FDIFF,samples=9,max_abs=" << fd_max_abs
            << ",max_rel=" << fd_max_abs/std::max(fd_ref_max,1e-300) << '\n';
  double q1ms = median_time_ms(q_center, reps);
  double qlocalms = median_time_ms(q_localq_style, reps);
  double q2ms = median_time_ms(q_split, reps);
  std::cout << std::fixed << std::setprecision(6)
            << "QTIME,center_original," << q1ms << "\n"
            << "QTIME,localq_style," << qlocalms << "\n"
            << "QTIME,edge_parallel," << q2ms << "\n";

  for (auto &m : methods) {
    m.launch(); CUDA_CHECK(cudaDeviceSynchronize());
    const double *ptr = (std::string(m.name)=="original_gather") ? grad_ref : grad;
    auto hv = copy_vector(const_cast<double*>(ptr), 3*g.n);
    auto err = compare(href,hv);
    double momentum = 0.0;
    for (int d=0; d<3; ++d) {
      double s=0.0; for (int i=0;i<g.n;++i) s += hv[3*i+d];
      momentum = std::max(momentum,std::abs(s));
    }
    double ms = median_time_ms(m.launch, reps);
    std::cout << "FORCE," << m.name << ",ms=" << ms << ",atomics=" << m.atomics
              << ",max_abs=" << std::scientific << std::get<0>(err)
              << ",max_rel=" << std::get<1>(err) << ",rms=" << std::get<2>(err)
              << ",sum_grad_max=" << momentum << std::fixed << '\n';
  }

  auto pipe_original = [&] {
    q_center();
    force_original_gather<L><<<atom_grid,block>>>(g.n,g.offsets,g.neighbors,g.dr,qlm,ql,grad_ref);
  };
  auto pipe_localq = [&] {
    q_localq_style();
    CUDA_CHECK(cudaMemsetAsync(grad,0,sizeof(double)*3*g.n));
    force_localq_atomic<L><<<atom_grid,block>>>(g.n,g.offsets,g.neighbors,g.dr,qlm2,ql2,grad);
  };
  auto pipe_localq_split = [&] {
    q_localq_style();
    CUDA_CHECK(cudaMemsetAsync(neighbor_grad,0,sizeof(double)*3*g.n));
    force_center_split<L><<<atom_grid,block>>>(g.n,g.offsets,g.neighbors,g.dr,qlm2,ql2,self_grad,neighbor_grad);
    combine_grad<<<value_grid,block>>>(g.n,self_grad,neighbor_grad,grad);
  };
  auto pipe_center_edge_atomic = [&] {
    q_center();
    CUDA_CHECK(cudaMemsetAsync(grad,0,sizeof(double)*3*g.n));
    force_edge_atomic<L><<<edge_grid,block>>>(g.n,g.e,g.sources,g.neighbors,g.offsets,g.dr,qlm,ql,grad);
  };
  auto pipe_center_edge_gather = [&] {
    q_center();
    force_edge_store<L><<<edge_grid,block>>>(g.n,g.e,g.sources,g.offsets,g.dr,qlm,ql,edge_value);
    force_reverse_gather<<<atom_grid,block>>>(g.n,g.offsets,g.incoming_offsets,g.incoming_edges,edge_value,grad);
  };
  auto pipe_edge_atomic = [&] {
    q_split();
    CUDA_CHECK(cudaMemsetAsync(grad,0,sizeof(double)*3*g.n));
    force_edge_atomic<L><<<edge_grid,block>>>(g.n,g.e,g.sources,g.neighbors,g.offsets,g.dr,qlm2,ql2,grad);
  };
  auto pipe_edge_gather = [&] {
    q_split();
    force_edge_store<L><<<edge_grid,block>>>(g.n,g.e,g.sources,g.offsets,g.dr,qlm2,ql2,edge_value);
    force_reverse_gather<<<atom_grid,block>>>(g.n,g.offsets,g.incoming_offsets,g.incoming_edges,edge_value,grad);
  };
  auto pipe_center_pair_atomic = [&] {
    q_center();
    CUDA_CHECK(cudaMemsetAsync(grad,0,sizeof(double)*3*g.n));
    force_pair_fused_atomic<L><<<pair_grid,block>>>(g.n,g.npairs,g.pair_i,g.pair_j,g.pair_dr,g.offsets,qlm,ql,grad);
  };
  auto pipe_center_pair_gather = [&] {
    q_center();
    force_pair_fused_store<L><<<pair_grid,block>>>(g.n,g.npairs,g.pair_i,g.pair_j,g.pair_dr,g.offsets,qlm,ql,pair_value);
    force_incidence_gather<<<atom_grid,block>>>(g.n,g.incidence_offsets,g.signed_incidence,pair_value,grad);
  };
  std::vector<std::pair<const char*,std::function<void()>>> pipelines{
      {"original",pipe_original},
      {"localq_style_atomic",pipe_localq},
      {"localq_style_center_split",pipe_localq_split},
      {"center_q_edge_atomic",pipe_center_edge_atomic},
      {"center_q_edge_gather",pipe_center_edge_gather},
      {"center_q_pair_fused_atomic",pipe_center_pair_atomic},
      {"center_q_pair_fused_gather",pipe_center_pair_gather},
      {"edge_q_edge_atomic",pipe_edge_atomic},
      {"edge_q_edge_gather",pipe_edge_gather}};
  for (auto &p : pipelines) {
    double ms = median_time_ms(p.second,reps);
    std::cout << "PIPELINE," << p.first << ",ms=" << std::fixed << std::setprecision(6) << ms << '\n';
  }

  cudaFree(qlm); cudaFree(qlm2); cudaFree(ylm); cudaFree(ql); cudaFree(ql2);
  cudaFree(grad_ref); cudaFree(grad); cudaFree(self_grad); cudaFree(neighbor_grad);
  cudaFree(edge_value); cudaFree(pair_value); cudaFree(dr_work);
}

int main(int argc, char **argv) {
  int reps = argc > 1 ? std::atoi(argv[1]) : 40;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  std::cout << "GPU," << prop.name << ",cc=" << prop.major << '.' << prop.minor
            << ",global_mem=" << prop.totalGlobalMem << '\n';
  std::cout << "BUILD," << __DATE__ << ' ' << __TIME__ << ",repetitions=" << reps << '\n';
  for (const auto &[nc, cutoff] : std::vector<std::pair<int,double>>{{11,0.82},{11,1.30},{16,0.82},{16,1.30}}) {
    HostGraph g = make_fcc_graph(nc,cutoff);
    int min_degree = 1e9, max_degree = 0;
    for (int i=0;i<g.n;++i) {
      int d=g.offsets[i+1]-g.offsets[i]; min_degree=std::min(min_degree,d); max_degree=std::max(max_degree,d);
    }
    std::cout << "GRAPH,nc=" << nc << ",cutoff=" << cutoff << ",N=" << g.n
              << ",E=" << g.neighbors.size() << ",degree_min=" << min_degree
              << ",degree_max=" << max_degree << '\n';
    run_case<4>(g,reps);
    run_case<6>(g,reps);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
