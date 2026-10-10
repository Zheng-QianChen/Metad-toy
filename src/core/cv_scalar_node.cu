#include "fix_crystallize.h"

MetaD_zqc::CVLeaf::CVLeaf(CV* cv, CVEval eval) : cv_(cv), eval_(eval) {}

double MetaD_zqc::CVLeaf::forward() {
    value_ = (cv_->*(eval_.forward))();
    return value_;
}

void MetaD_zqc::CVLeaf::backward(double du_dc) {
    // Same CV object may be bound to two SYMBOL modes. Write this leaf's
    // forward scalar back before bias_force / get_dcvdx read cv_value.
    // bias_force then applies du/dx = du_dc * dc/dx and the virial tally.
    cv_->set_cv_value(value_);
    (cv_->*(eval_.backward))(du_dc);
}
