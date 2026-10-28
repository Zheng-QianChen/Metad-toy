#pragma once
#include "zqc_engine.h"

namespace MetaD_zqc {
// DISTANCE 的引擎无关计算状态。ID 长期固定，owned 下标与位移每次 forward 重建。
// Host 全局标量节点使用此对象；解析、输出格式由各宿主的 CV 包装器负责。
class DistanceCalculation {
    AtomId atom_ids_[2];
    int owned_indices_[2] = {-1, -1};
    double displacement_[3] = {};
    double value_ = 0;
    std::uint64_t evaluation_id_ = 0;
    const EngineAdapter* engine_ = nullptr;
public:
    DistanceCalculation(AtomId first, AtomId second) : atom_ids_{first, second} {
        if (!first || !second || first == second)
            throw std::invalid_argument("DISTANCE requires two distinct positive atom IDs");
    }
    double forward(EngineAdapter& engine) {
        engine_ = nullptr;
        if (!engine.evaluation_id()) throw std::logic_error("DISTANCE: evaluation has not started");
        double coordinates[8] = {};
        engine.read_owned_atoms(atom_ids_, 2, coordinates, owned_indices_);
        engine.sum(coordinates, 8);
        if (coordinates[3] != 1 || coordinates[7] != 1)
            throw std::runtime_error("DISTANCE: each atom ID must have exactly one owner");
        for (int axis = 0; axis < 3; ++axis)
            displacement_[axis] = coordinates[4+axis] - coordinates[axis];
        engine.minimum_image(displacement_);
        value_ = std::sqrt(displacement_[0]*displacement_[0] + displacement_[1]*displacement_[1] +
                           displacement_[2]*displacement_[2]);
        if (!(value_ > 0) || !std::isfinite(value_))
            throw std::runtime_error("DISTANCE: zero or non-finite separation has no defined gradient");
        evaluation_id_ = engine.evaluation_id();
        engine_ = &engine;
        return value_;
    }
    void backward(EngineAdapter& engine, double seed) const {
        if (engine_ != &engine || evaluation_id_ != engine.evaluation_id())
            throw std::logic_error("DISTANCE: forward and backward belong to different evaluations");
        if (!std::isfinite(seed)) throw std::invalid_argument("DISTANCE: non-finite adjoint");
        const double scale = seed/value_;
        const double* delta = displacement_;
        // W=-dV/d(epsilon)。每个 owned 端点保存一半，跨 rank 的两端合计为完整 pair virial。
        const double half_virial[6] = {
            -0.5*scale*delta[0]*delta[0], -0.5*scale*delta[1]*delta[1], -0.5*scale*delta[2]*delta[2],
            -0.5*scale*delta[0]*delta[1], -0.5*scale*delta[0]*delta[2], -0.5*scale*delta[1]*delta[2]};
        for (int endpoint = 0; endpoint < 2; ++endpoint) {
            if (owned_indices_[endpoint] < 0) continue;
            const double sign = endpoint == 0 ? 1.0 : -1.0;
            double force[3];
            for (int axis = 0; axis < 3; ++axis) force[axis] = sign*scale*delta[axis];
            engine.add_owned_force_virial(owned_indices_[endpoint], force,
                {{half_virial,1,6,6,1,6,MemorySpace::Host}, VirialRepresentation::Symmetric6});
        }
    }
};
} // namespace MetaD_zqc
