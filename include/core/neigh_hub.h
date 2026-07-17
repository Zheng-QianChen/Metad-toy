#pragma once

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {
class Fix;
class LAMMPS;
class NeighList;
}    // namespace LAMMPS_NS

namespace MetaD_zqc {

// Request-side fields aligned with NeighRequest::identical() (requestor-set params).
// skip arrays are not supported here (skip must stay 0).
struct NeighSpec {
  int full = 1;             // 1=full, 0=half
  int ghost = 0;
  int newton = 0;
  int size_flag = 0;
  int history = 0;
  int granonesided = 0;
  int respainner = 0;
  int respamiddle = 0;
  int respaouter = 0;
  int bond = 0;
  int omp = 0;
  int intel = 0;
  int kokkos_host = 0;
  int kokkos_device = 0;
  int ssa = 0;
  int skip = 0;
  int use_cutoff = 0;       // NeighRequest::cut
  double cutoff = 0.0;
  // -1 = auto: occasional unless ghost (LAMMPS BIN forbids occasional+ghost)
  int occasional = -1;
};

// Owns Fix neighbor requests: dedupe by key, distribute NeighList* by neigh_id.
class NeighHub {
 public:
  NeighHub() = default;

  void bind(LAMMPS_NS::LAMMPS *lmp_in, LAMMPS_NS::Fix *fix_in);

  static double canonicalize_cutoff(double c);
  static long long cutoff_key_token(double c_use);

  // Cold path (CV factory): request once per unique spec, return neigh_id (>=1).
  int get_or_create(NeighSpec spec);

  void on_init_list(int id, LAMMPS_NS::NeighList *ptr);

  // Call from Fix::init() every run segment (not only first_run).
  // Forces next ensure() to rebuild / rebind after write_restart + new run.
  void invalidate_all();

  // Hot path: ensure list is built, then O(1) pointer fetch.
  void ensure(int id);
  void ensure_all();
  LAMMPS_NS::NeighList *list(int id) const;

  int max_id() const { return next_id_ - 1; }

 private:
  std::string make_key(const NeighSpec &s) const;
  LAMMPS_NS::NeighList *resolve_list(int id);

  LAMMPS_NS::LAMMPS *lmp_ = nullptr;
  LAMMPS_NS::Fix *fix_ = nullptr;
  int next_id_ = 1;
  std::unordered_map<std::string, int> key_to_id_;
  std::vector<LAMMPS_NS::NeighList *> lists_;       // [0] unused
  std::vector<unsigned char> occasional_;           // parallel to lists_
  std::vector<long long> ensured_lastcall_;          // neighbor->lastcall at last ensure
};

}    // namespace MetaD_zqc
