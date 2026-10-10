// Host check for the single-CV node contract.
// Does not link LAMMPS. CVLeaf in the plugin is this same forward-then-backward
// shape: one scalar in, du/dc in, dc/dx stays inside the leaf.

#include "cv_scalar_node.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int fails = 0;

void expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL " << msg << "\n";
        ++fails;
    }
}

// Stand-in for one base CV. backward records du/dc; dc/dx is applied by the
// caller that models get_dcvdx, matching CVLeaf (the leaf does not compose).
struct ScalarLeaf : MetaD_zqc::CVScalarNode {
    double x = 0.0;
    double du_dc = 0.0;
    int forwards = 0;
    int backwards = 0;

    double forward() override {
        value_ = x;
        ++forwards;
        return value_;
    }

    void backward(double upstream) override {
        du_dc = upstream;
        ++backwards;
    }
};

// One-input wrapper used only in this test. c = k * child, dc/dx = k.
// Shows a later single-CV outer function can sit on one leaf. Not registered
// in the LAMMPS plugin.
struct ScaleNode : MetaD_zqc::CVScalarNode {
    MetaD_zqc::CVScalarNode* child;
    double k;

    ScaleNode(MetaD_zqc::CVScalarNode* child_in, double k_in)
        : child(child_in), k(k_in) {}

    double forward() override {
        value_ = k * child->forward();
        return value_;
    }

    void backward(double du_dc) override {
        child->backward(du_dc * k);
    }
};

void test_split_names() {
    std::string main_func;
    std::string sub_param;
    MetaD_zqc::split_cv_func_name("AVE", main_func, sub_param);
    expect(main_func == "AVE" && sub_param.empty(), "AVE has no sub-parameter");

    MetaD_zqc::split_cv_func_name("SW_FUNC.Fermi", main_func, sub_param);
    expect(main_func == "SW_FUNC" && sub_param == "Fermi", "SW_FUNC.Fermi splits");

    MetaD_zqc::split_cv_func_name("MEAN_SOLID.Cubic", main_func, sub_param);
    expect(main_func == "MEAN_SOLID" && sub_param == "Cubic", "MEAN_SOLID.Cubic splits");
}

void test_identity_leaf_chain() {
    ScalarLeaf leaf;
    leaf.x = 1.25;
    const double c = leaf.forward();
    expect(c == 1.25 && leaf.value() == 1.25, "leaf forward stores the scalar");
    // Expression S = v => dS/dv = 1, so backward sees dV/dc unchanged.
    const double dVdc = 4.0;
    const double dS_dv = 1.0;
    leaf.backward(dVdc * dS_dv);
    expect(leaf.du_dc == 4.0, "identity chain passes du/dc through");
    expect(leaf.forwards == 1 && leaf.backwards == 1, "one forward then one backward");
}

void test_two_modes_keep_own_scalar() {
    // Two SYMBOL modes can share one CV object. Each leaf must backward
    // with the scalar from its own forward, not the other mode's value.
    double shared_cv = 0.0;
    struct ModeLeaf : MetaD_zqc::CVScalarNode {
        double* shared;
        double x;
        double seen_cv = -1.0;
        explicit ModeLeaf(double* shared_in, double x_in) : shared(shared_in), x(x_in) {}
        double forward() override {
            value_ = x;
            *shared = x;
            return value_;
        }
        void backward(double) override { seen_cv = value_; }
    };
    ModeLeaf first(&shared_cv, 2.0);
    ModeLeaf second(&shared_cv, 5.0);
    first.forward();
    second.forward();
    expect(shared_cv == 5.0, "later forward overwrites the shared scalar");
    first.backward(3.0);
    expect(first.seen_cv == 2.0, "first leaf still has its own forward value");
    second.backward(3.0);
    expect(second.seen_cv == 5.0, "second leaf still has its own forward value");
}

void test_single_input_scale_chain() {
    // Analytic: c = 2.5 * x, x = 4, du/dc = 0.5 => du/dx = 1.25.
    ScalarLeaf leaf;
    leaf.x = 4.0;
    ScaleNode scaled(&leaf, 2.5);
    const double c = scaled.forward();
    expect(std::fabs(c - 10.0) < 1e-12, "scale forward c = k * x");
    scaled.backward(0.5);
    expect(std::fabs(leaf.du_dc - 1.25) < 1e-12, "scale chain du/dx = du/dc * k");
}

}  // namespace

int main() {
    test_split_names();
    test_identity_leaf_chain();
    test_two_modes_keep_own_scalar();
    test_single_input_scale_chain();
    if (fails != 0) {
        std::cerr << fails << " check(s) failed\n";
        return 1;
    }
    std::cout << "scalar_node_dispatch_test ok\n";
    return 0;
}
