#pragma once  // 必须添加这一行

#include <unordered_map>
#include <vector>

#include "lammps.h"
#include "memory.h"
#include "fix_crystallize.h"
// #include "exprtk.hpp"

namespace MetaD_zqc {
    class Gaussian_Hill_Base{
        friend class LAMMPS_NS::FixMetadynamics;
        
        protected:
        LAMMPS_NS::LAMMPS *lmp;
        FILE* f_hills;
        FILE* f_check;
        double sigma, height0, biasf, kBT;
        double KB;
        double current_temp=300.0;
        // HILLS
        int continue_from_file;
        // setting: "mode"
        int settings;
        // dim: dimention
        int cv_dim;
        // WellT_bool: one of mode
        int WellT_bool;

        public:
        ~Gaussian_Hill_Base();
        Gaussian_Hill_Base(LAMMPS_NS::LAMMPS *lmp, FILE* f_check,
                            int cv_dim,
                            double sigma, double height0, double biasf,
                            int continue_from_file, int WellT_bool);
        virtual void io_hills() {};
        virtual void init_set_mode() {};
        virtual void add_hill(double *) {};
        virtual void get_dVdcv(double *, double *) {};
        virtual void write_hill(double *cv_values) {};
        // 当前 CV 处的偏置势 V_b(s)，供 Fix thermo/energy 记账
        virtual double get_bias_energy(double * /*cv_values*/) { return 0.0; }
    };

    // =========================================================================
    // GH_t0_uniformGrid: add gaussian hill to grid, 
    // and calculate bias and gradient from grid
    // =========================================================================

    template<int D>
    class GH_t0_uniformGrid: public Gaussian_Hill_Base{
        private:
        int *nbin, *cvspace_loc;
        int grid_size;
        // gaussian hills set
        double *cv_bound, *dcv;
        double *bias_grid;
        // add_to_grid
        double *delta_x;
        int *index_radius;
        int *lower, *upper;
        
        void add_to_grid(double *cv_values, double w, double sig);
        void get_cvspace_loc(double* cv_values, int* cvspace_loc);

        public:
        ~GH_t0_uniformGrid();
        GH_t0_uniformGrid(LAMMPS_NS::LAMMPS *lmp, FILE* f_check,
                            int cv_dim,
                            double sigma, double height0, double biasf,
                            int continue_from_file, int WellT_bool,
                            double *cv_bound, int *nbin);
        void init_hills();
        void init_set_mode() override;
        void add_hill(double *cv_values) override;
        void get_dVdcv(double *, double *) override;
        void write_hill(double *cv_values, double w);
        double get_total_bias(int* cvspace_loc);
        double get_bias_energy(double *cv_values) override;
        double gauss_calc(int dim, double* dx, double s);
    };
    
    template class GH_t0_uniformGrid<1>;
    template class GH_t0_uniformGrid<2>;
    template class GH_t0_uniformGrid<3>;
    template<> void MetaD_zqc::GH_t0_uniformGrid<1>::add_to_grid(double *cv_values, double w, double sig);
    template<> void MetaD_zqc::GH_t0_uniformGrid<2>::add_to_grid(double *cv_values, double w, double sig);
    template<> void MetaD_zqc::GH_t0_uniformGrid<3>::add_to_grid(double *cv_values, double w, double sig);
    template<> void MetaD_zqc::GH_t0_uniformGrid<1>::get_dVdcv(double *cv_values, double *dVdcvs);
    template<> void MetaD_zqc::GH_t0_uniformGrid<2>::get_dVdcv(double *cv_values, double *dVdcvs);
    template<> void MetaD_zqc::GH_t0_uniformGrid<3>::get_dVdcv(double *cv_values, double *dVdcvs);
    template<> double MetaD_zqc::GH_t0_uniformGrid<1>::get_total_bias(int* cvspace_loc);
    template<> double MetaD_zqc::GH_t0_uniformGrid<2>::get_total_bias(int* cvspace_loc);
    template<> double MetaD_zqc::GH_t0_uniformGrid<3>::get_total_bias(int* cvspace_loc);
    template<> double MetaD_zqc::GH_t0_uniformGrid<1>::get_bias_energy(double *cv_values);
    template<> double MetaD_zqc::GH_t0_uniformGrid<2>::get_bias_energy(double *cv_values);
    template<> double MetaD_zqc::GH_t0_uniformGrid<3>::get_bias_energy(double *cv_values);
    template<> double GH_t0_uniformGrid<1>::gauss_calc(int dim, double* dx, double s);
    template<> double GH_t0_uniformGrid<2>::gauss_calc(int dim, double* dx, double s);
    template<> double GH_t0_uniformGrid<3>::gauss_calc(int dim, double* dx, double s);

    // =========================================================================
    // GH_t1_sparseHash: add gaussian hill to grid, 
    // and calculate bias and gradient from grid
    // =========================================================================


    template<int D>
    class GH_t1_sparseHash : public Gaussian_Hill_Base {
    private:
        // 哈希表：Key 是多维坐标数组，Value 是该点的势能值
        // 注意：为了 LAMMPS 插件兼容性，建议使用简单类型的 Key
        struct CoordKey{
            int c[D];
            bool operator==(const CoordKey& other) const {
                for(int i=0; i<D; ++i) if(c[i] != other.c[i]) return false;
                return true;
            }
        };

        struct CoordHash {
            size_t operator()(const CoordKey& k) const {
                size_t h = 0;
                for(int i=0; i<D; ++i) h ^= std::hash<int>{}(k.c[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
                return h;
            }
        };

        struct GridIndexHash {
            int cv_dim;
            const int* nbin;
            GridIndexHash(int d, const int* n) : cv_dim(d), nbin(n) {}

            size_t operator()(const std::vector<int>& coords) const {
                size_t h = 0;
                for (int i = 0; i < cv_dim; ++i) {
                    h = h * 31 + coords[i]; // 简单的质数扰动哈希
                }
                return h;
            }
        };
        struct Entry { CoordKey k; double v; };

        std::unordered_map<CoordKey, double, CoordHash> bias_hash;

        int *nbin, *cvspace_loc;
        double *cv_bound, *dcv;
        // double *delta_x;
        // double *p; // 用于计算高斯函数的临时数组
        // int *base_coord;
        // double *frac;
        // CoordKey key;
        // int *lower, *upper;
        int *index_radius;

        void io_hills();
        void add_to_grid(double *cv_values, double w);
        void recursive_add2grid(int dim, CoordKey& key, 
                                double* dx_array,
                                double w, double* cv_values);
        void GridHashBcast();

    public:
        GH_t1_sparseHash(LAMMPS_NS::LAMMPS *lmp, FILE* f_check,
                         int cv_dim, double sigma, double height0, double biasf,
                         int continue_from_file, int WellT_bool,
                         double *cv_bound, int *nbin);
        
        ~GH_t1_sparseHash();

        void init_set_mode() override;
        void write_hill(double *cv_values, double w);
        void add_hill(double *cv_values) override;

        void get_dVdcv(double *cv_values, double *dVdcvs) override;
        double mixed_recursive_logic(int current_dim, int target_dim, 
                                                int* coord, double* frac);
        
        // 核心：获取或初始化哈希点
        double get_bias_at(const CoordKey& k);
        double get_total_bias(int* cvspace_loc);
        double get_bias_energy(double *cv_values) override;

        // 辅助：CV值转网格坐标
        void get_cvspace_loc(double* cv, int* coord);

        double gauss_calc(int dim, double* dx, double s);
    };
    template class GH_t1_sparseHash<1>;
    template class GH_t1_sparseHash<2>;
    template class GH_t1_sparseHash<3>;


    // =========================================================================
    // GH_t2_inrun: do not add gaussian hill to grid
    // =========================================================================


    class GH_t2_inrun : public Gaussian_Hill_Base {
        private:
        public:
    };

}