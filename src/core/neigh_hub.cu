#include "neigh_hub.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "lammps.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <cstdio>
#include <sstream>

using namespace LAMMPS_NS;
using namespace MetaD_zqc;

namespace {
constexpr double kCutQuantum = 1e-9;
}

void NeighHub::bind(LAMMPS *lmp_in, Fix *fix_in)
{
  lmp_ = lmp_in;
  fix_ = fix_in;
  specs_.assign(1, NeighSpec{});
  lists_.assign(1, nullptr);
  occasional_.assign(1, 0);
  ensured_lastcall_.assign(1, -1);
}

double NeighHub::canonicalize_cutoff(double c)
{
  return std::round(c / kCutQuantum) * kCutQuantum;
}

long long NeighHub::cutoff_key_token(double c_use)
{
  return std::llround(c_use / kCutQuantum);
}

std::string NeighHub::make_key(const NeighSpec &s) const
{
  // Field order mirrors NeighRequest::identical() requestor-set flags.
  std::ostringstream oss;
  oss << s.full << '_' << (1 - s.full) << '_'
      << s.occasional << '_' << s.newton << '_' << s.ghost << '_'
      << s.size_flag << '_' << s.history << '_' << s.granonesided << '_'
      << s.respainner << '_' << s.respamiddle << '_' << s.respaouter << '_'
      << s.bond << '_' << s.omp << '_' << s.intel << '_'
      << s.kokkos_host << '_' << s.kokkos_device << '_' << s.ssa << '_'
      << s.skip << '_' << s.use_cutoff << '_'
      << (s.use_cutoff ? cutoff_key_token(s.cutoff) : 0LL);
  return oss.str();
}

void NeighHub::normalize_spec(NeighSpec &spec) const
{
  if (spec.skip != 0)
    lmp_->error->all(FLERR, "NeighHub does not support skip neighbor requests");

  // LAMMPS BIN: occasional + ghost is forbidden → force perpetual for ghost.
  if (spec.occasional < 0)
    spec.occasional = spec.ghost ? 0 : 1;
  if (spec.ghost && spec.occasional)
    spec.occasional = 0;

  if (spec.use_cutoff) {
    if (!(spec.cutoff > 0.0) || !std::isfinite(spec.cutoff))
      lmp_->error->all(FLERR, "NeighHub: invalid neighbor cutoff");
    spec.cutoff = canonicalize_cutoff(spec.cutoff);
  } else {
    spec.cutoff = 0.0;
  }
}

NeighRequest *NeighHub::issue_request(int id, const NeighSpec &spec)
{
  int flags = NeighConst::REQ_DEFAULT;
  if (spec.full) flags |= NeighConst::REQ_FULL;
  if (spec.ghost) flags |= NeighConst::REQ_GHOST;
  if (spec.occasional) flags |= NeighConst::REQ_OCCASIONAL;
  if (spec.size_flag) flags |= NeighConst::REQ_SIZE;
  if (spec.history) flags |= NeighConst::REQ_HISTORY;
  if (spec.newton == 1) flags |= NeighConst::REQ_NEWTON_ON;
  if (spec.newton == 2) flags |= NeighConst::REQ_NEWTON_OFF;
  if (spec.ssa) flags |= NeighConst::REQ_SSA;
  if (spec.respainner && spec.respaouter && !spec.respamiddle)
    flags |= NeighConst::REQ_RESPA_INOUT;
  if (spec.respainner && spec.respamiddle && spec.respaouter)
    flags |= NeighConst::REQ_RESPA_ALL;

  NeighRequest *req = lmp_->neighbor->add_request(fix_, flags);
  req->set_id(id);
  if (spec.use_cutoff) req->set_cutoff(spec.cutoff);
  if (spec.intel) req->enable_intel();
  if (spec.kokkos_host) req->set_kokkos_host(1);
  if (spec.kokkos_device) req->set_kokkos_device(1);
  return req;
}

int NeighHub::get_or_create(NeighSpec spec)
{
  if (!lmp_ || !fix_) {
    fprintf(stderr, "NeighHub::get_or_create called before bind()\n");
    return -1;
  }

  normalize_spec(spec);

  const std::string key = make_key(spec);
  auto it = key_to_id_.find(key);
  if (it != key_to_id_.end()) return it->second;

  // Only register here. Actual add_request happens in rerequest_all() from
  // Fix::init(), so each System init (run / write_restart) gets a fresh request
  // and we never double-add before the first Neighbor::init.
  const int id = next_id_++;
  if ((int)lists_.size() <= id) {
    specs_.resize(id + 1);
    lists_.resize(id + 1, nullptr);
    occasional_.resize(id + 1, 0);
    ensured_lastcall_.resize(id + 1, -1);
  }
  specs_[id] = spec;
  occasional_[id] = spec.occasional ? 1 : 0;
  key_to_id_[key] = id;

  if (lmp_->comm->me == 0) {
    fprintf(stderr,
            "[NeighHub] register id=%d key=%s occasional=%d ghost=%d cut=%g "
            "(add_request deferred to Fix::init / rerequest_all)\n",
            id, key.c_str(), (int)occasional_[id], spec.ghost,
            spec.use_cutoff ? spec.cutoff : 0.0);
  }
  return id;
}

void NeighHub::rerequest_all()
{
  if (!lmp_ || !fix_) return;
  for (int id = 1; id < (int)specs_.size(); ++id) {
    issue_request(id, specs_[id]);
    lists_[id] = nullptr;
    ensured_lastcall_[id] = -1;
  }
  if (lmp_->comm->me == 0 && specs_.size() > 1) {
    fprintf(stderr, "[NeighHub] rerequest_all: %d id(s)\n",
            (int)specs_.size() - 1);
  }
}

void NeighHub::on_init_list(int id, NeighList *ptr)
{
  if (id <= 0) return;
  if ((int)lists_.size() <= id) {
    specs_.resize(id + 1);
    lists_.resize(id + 1, nullptr);
    occasional_.resize(id + 1, 0);
    ensured_lastcall_.resize(id + 1, -1);
  }
  lists_[id] = ptr;
  // Neighbor re-init (new run / write_restart path): do not trust old pages.
  ensured_lastcall_[id] = -1;
}

void NeighHub::invalidate_all()
{
  for (size_t i = 0; i < ensured_lastcall_.size(); ++i)
    ensured_lastcall_[i] = -1;
}

NeighList *NeighHub::resolve_list(int id)
{
  // Prefer live pointer from Neighbor::lists[] matching request id.
  // Avoids dangling NeighList* after Neighbor rebuilds between run segments.
  if (!lmp_ || !lmp_->neighbor) return nullptr;
  NeighList **lists = lmp_->neighbor->lists;
  const int nlist = lmp_->neighbor->nlist;
  for (int i = 0; i < nlist; ++i) {
    if (lists[i] && lists[i]->id == id) {
      lists_[id] = lists[i];
      return lists[i];
    }
  }
  // Fallback to cached pointer (may still be valid if init_list just ran)
  if (id > 0 && id < (int)lists_.size()) return lists_[id];
  return nullptr;
}

void NeighHub::ensure(int id)
{
  if (!lmp_ || id <= 0 || id >= (int)lists_.size())
    lmp_->error->all(FLERR, "NeighHub::ensure: invalid neigh_id");

  NeighList *nl = resolve_list(id);
  if (nl == nullptr)
    lmp_->error->all(FLERR, "NeighHub::ensure: cannot resolve NeighList for id");

  if (occasional_[id]) {
    // Occasional: build_one is the only legal fill path.
    lmp_->neighbor->build_one(nl);
    nl = resolve_list(id);
  } else {
    // Perpetual (LocalQ ghost): NEVER call Neighbor::build() from post_force.
    // Mid-force full rebuild corrupts pair (MEAM) lists → NaN positions / insane Press.
    // Trust Verlet/Neighbor to fill perpetual lists; we only rebind + validate.
    nl = resolve_list(id);
  }

  if (nl == nullptr || nl->numneigh == nullptr || nl->firstneigh == nullptr ||
      nl->ilist == nullptr)
    lmp_->error->all(FLERR,
                     "NeighHub::ensure: perpetual/occasional list has null "
                     "numneigh/firstneigh/ilist (Neighbor not ready?)");
  if ((nl->inum + nl->gnum) <= 0 && lmp_->atom->nlocal > 0)
    lmp_->error->all(FLERR,
                     "NeighHub::ensure: neighbor list empty (inum+gnum==0)");

  lists_[id] = nl;
  ensured_lastcall_[id] = (long long)lmp_->neighbor->lastcall;
}

void NeighHub::ensure_all()
{
  // Ensure every registered id, even if cached NeighList* is temporarily null
  // (resolve_list rebinds from Neighbor::lists[]).
  for (int id = 1; id < (int)lists_.size(); ++id) ensure(id);
}

NeighList *NeighHub::list(int id) const
{
  if (id <= 0 || id >= (int)lists_.size()) return nullptr;
  return lists_[id];
}
