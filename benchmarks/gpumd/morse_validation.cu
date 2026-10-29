/*----------------------------------------------------------------------------80
O₂ Morse 校验势（validation-only）：两原子最小镜像 + FP64 对势。
公式（截断内）：U = D * (1 - exp(-α*(r-r₀)))² - D；截断处及以外为零。
------------------------------------------------------------------------------*/

#include "morse_validation.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstring>

MetaDToyMorseValidation::MetaDToyMorseValidation(FILE* input, int num_types, int num_atoms)
{
  if (num_types != 1) {
    PRINT_INPUT_ERROR("MetaDToyMorseValidation requires num_types == 1.\n");
  }
  if (num_atoms != 2) {
    PRINT_INPUT_ERROR("MetaDToyMorseValidation requires num_atoms == 2.\n");
  }

  char element_symbol[16];
  int count = fscanf(input, "%15s", element_symbol);
  PRINT_SCANF_ERROR(count, 1, "Reading error for MetaDToyMorseValidation element symbol.");
  if (strcmp(element_symbol, "O") != 0) {
    PRINT_INPUT_ERROR("MetaDToyMorseValidation fixture expects element symbol O.\n");
  }

  count = fscanf(input, "%lf%lf%lf%lf", &depth_eV, &alpha_inverse_A, &equilibrium_A, &cutoff_A);
  PRINT_SCANF_ERROR(count, 4, "Reading error for MetaDToyMorseValidation Morse parameters.");

  // 四个参数均须为有限正数（D、α、r₀、截断）。
  if (!(std::isfinite(depth_eV) && depth_eV > 0.0)) {
    PRINT_INPUT_ERROR("MetaDToyMorseValidation depth_eV must be finite and positive.\n");
  }
  if (!(std::isfinite(alpha_inverse_A) && alpha_inverse_A > 0.0)) {
    PRINT_INPUT_ERROR("MetaDToyMorseValidation alpha_inverse_A must be finite and positive.\n");
  }
  if (!(std::isfinite(equilibrium_A) && equilibrium_A > 0.0)) {
    PRINT_INPUT_ERROR("MetaDToyMorseValidation equilibrium_A must be finite and positive.\n");
  }
  if (!(std::isfinite(cutoff_A) && cutoff_A > 0.0)) {
    PRINT_INPUT_ERROR("MetaDToyMorseValidation cutoff_A must be finite and positive.\n");
  }

  rc = cutoff_A;
  printf(
    "Use MetaDToyMorseValidation (O2 fixture): D=%g eV, alpha=%g 1/A, r0=%g A, cutoff=%g A\n",
    depth_eV,
    alpha_inverse_A,
    equilibrium_A,
    cutoff_A);
}

MetaDToyMorseValidation::~MetaDToyMorseValidation(void)
{
}

// 单线程核：原子 0/1 的 SoA 位置 → 最小镜像 δ → Morse U 与 U' → 累加势/力/Full9 virial。
// 位置/力 SoA：分量跨度为原子数 N，即 x[i], y[i]=pos[N+i], z[i]=pos[2N+i]。
// Virial Full9 SoA 顺序：xx yy zz xy xz yz yx zx zy，跨度同为 N。
static __global__ void gpu_morse_o2_pair(
  const Box box,
  const int number_of_atoms,
  const double depth_eV,
  const double alpha_inverse_A,
  const double equilibrium_A,
  const double cutoff_A,
  const double* __restrict__ g_position,
  double* g_potential,
  double* g_force,
  double* g_virial)
{
  // 校验夹具仅两原子；单线程完成整对计算，避免设备级同步。
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  const double* g_x = g_position;
  const double* g_y = g_position + number_of_atoms;
  const double* g_z = g_position + number_of_atoms * 2;

  // δ = r₁ - r₀（最小镜像后）；力 F₀ = +U'(r)·δ/r，F₁ = -F₀。
  double delta_x = g_x[1] - g_x[0];
  double delta_y = g_y[1] - g_y[0];
  double delta_z = g_z[1] - g_z[0];
  apply_mic(box, delta_x, delta_y, delta_z);

  const double pair_distance =
    sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);

  // 截断处及以外：能量、力、virial 贡献均为零。
  if (!(pair_distance < cutoff_A)) {
    return;
  }

  // U = D*(1-exp(-α*(r-r₀)))² - D
  const double exponential_term = exp(-alpha_inverse_A * (pair_distance - equilibrium_A));
  const double one_minus_exponential = 1.0 - exponential_term;
  const double pair_energy =
    depth_eV * one_minus_exponential * one_minus_exponential - depth_eV;

  // 每原子各记一半势能；保留既有累加器（+=）。
  const double half_pair_energy = 0.5 * pair_energy;
  g_potential[0] += half_pair_energy;
  g_potential[1] += half_pair_energy;

  // r=0 时无法构造 δ/r；夹具正常键长下不会触发。
  if (!(pair_distance > 0.0)) {
    return;
  }

  // U' = 2·D·(1-exp(-α*(r-r₀)))·α·exp(-α*(r-r₀))
  const double energy_derivative =
    2.0 * depth_eV * one_minus_exponential * alpha_inverse_A * exponential_term;
  const double force_scale = energy_derivative / pair_distance;

  const double force_on_atom0_x = force_scale * delta_x;
  const double force_on_atom0_y = force_scale * delta_y;
  const double force_on_atom0_z = force_scale * delta_z;

  double* g_fx = g_force;
  double* g_fy = g_force + number_of_atoms;
  double* g_fz = g_force + number_of_atoms * 2;

  g_fx[0] += force_on_atom0_x;
  g_fy[0] += force_on_atom0_y;
  g_fz[0] += force_on_atom0_z;
  g_fx[1] -= force_on_atom0_x;
  g_fy[1] -= force_on_atom0_y;
  g_fz[1] -= force_on_atom0_z;

  // 每端半对 virial：½·(-U'/r)·(δ⊗δ)；Full9 顺序 xx yy zz xy xz yz yx zx zy。
  const double half_virial_scale = -0.5 * force_scale;
  const double virial_xx = half_virial_scale * delta_x * delta_x;
  const double virial_yy = half_virial_scale * delta_y * delta_y;
  const double virial_zz = half_virial_scale * delta_z * delta_z;
  const double virial_xy = half_virial_scale * delta_x * delta_y;
  const double virial_xz = half_virial_scale * delta_x * delta_z;
  const double virial_yz = half_virial_scale * delta_y * delta_z;
  const double virial_yx = half_virial_scale * delta_y * delta_x;
  const double virial_zx = half_virial_scale * delta_z * delta_x;
  const double virial_zy = half_virial_scale * delta_z * delta_y;

  for (int atom = 0; atom < 2; ++atom) {
    g_virial[atom + 0 * number_of_atoms] += virial_xx;
    g_virial[atom + 1 * number_of_atoms] += virial_yy;
    g_virial[atom + 2 * number_of_atoms] += virial_zz;
    g_virial[atom + 3 * number_of_atoms] += virial_xy;
    g_virial[atom + 4 * number_of_atoms] += virial_xz;
    g_virial[atom + 5 * number_of_atoms] += virial_yz;
    g_virial[atom + 6 * number_of_atoms] += virial_yx;
    g_virial[atom + 7 * number_of_atoms] += virial_zx;
    g_virial[atom + 8 * number_of_atoms] += virial_zy;
  }
}

void MetaDToyMorseValidation::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  const int number_of_atoms = type.size();
  // 单线程核；GPU_CHECK_KERNEL 在非 STRONG_DEBUG 下仅查 launch 错误，不设备同步。
  gpu_morse_o2_pair<<<1, 1>>>(
    box,
    number_of_atoms,
    depth_eV,
    alpha_inverse_A,
    equilibrium_A,
    cutoff_A,
    position.data(),
    potential.data(),
    force.data(),
    virial.data());
  GPU_CHECK_KERNEL
}
