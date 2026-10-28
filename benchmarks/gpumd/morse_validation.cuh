/*----------------------------------------------------------------------------80
仅用于 O₂ Morse 夹具验证的 GPUMD Potential 子类（非通用生产势）。
通过 ZQC_GPUMD_VALIDATION_MORSE 构建选项接入。
------------------------------------------------------------------------------*/

#pragma once
#include "force/potential.cuh"
#include <stdio.h>

// 校验专用：单元素、恰好两原子的移位 Morse 势。
// 输入顺序与 O₂ 夹具一致：元素符号 + depth_eV alpha_inverse_A equilibrium_A cutoff_A。
class MetaDToyMorseValidation : public Potential
{
public:
  using Potential::compute;

  MetaDToyMorseValidation(FILE* input, int num_types, int num_atoms);
  virtual ~MetaDToyMorseValidation(void);

  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

private:
  double depth_eV;          // D：势阱深度（eV）
  double alpha_inverse_A;   // α：宽度参数（Å⁻¹）
  double equilibrium_A;     // r₀：平衡键长（Å）
  double cutoff_A;          // 截断距离（Å）；r≥截断时能量/力/virial 均为零
};
