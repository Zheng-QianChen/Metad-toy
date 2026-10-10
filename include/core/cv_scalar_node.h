#pragma once

#include <string>

// Single-scalar CV node.
//
// One node owns one value. forward() writes that scalar. backward(du_dc)
// pushes du/dx = (du/dc) * (dc/dx) into this node only. A leaf's dc/dx is
// the CV's own get_dcvdx; callers pass du/dc in and do not apply forces
// themselves.
//
// This layer is one CV. It does not compose cv1, cv2, ... .

namespace MetaD_zqc {

class CVScalarNode {
public:
  virtual ~CVScalarNode() = default;

  // Compute this node's scalar and store it.
  virtual double forward() = 0;

  // Accept du/dc from the caller and apply this node's dc/dx.
  virtual void backward(double du_dc) = 0;

  double value() const { return value_; }

protected:
  double value_ = 0.0;
};

// "SW_FUNC.Fermi" -> main "SW_FUNC", sub "Fermi". No dot -> sub empty.
inline void split_cv_func_name(const std::string& func_name,
                               std::string& main_func,
                               std::string& sub_param) {
  const size_t dot_pos = func_name.find('.');
  if (dot_pos == std::string::npos) {
    main_func = func_name;
    sub_param.clear();
    return;
  }
  main_func = func_name.substr(0, dot_pos);
  sub_param = func_name.substr(dot_pos + 1);
}

}  // namespace MetaD_zqc
