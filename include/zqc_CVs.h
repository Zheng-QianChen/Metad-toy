#include "fix_crystallize.h"
#include "lammps.h"
#include "pair.h"
#include "neigh_request.h"

#define PI 3.1415926535897932385
namespace MetaD_zqc {
    // atoms distance
    class Distance : public CV {
    private:
        LAMMPS_NS::bigint atom_id1, atom_id2;
        bool pbc_x, pbc_y, pbc_z;
        double box_x, box_y, box_z;
        double dx, dy, dz;
    public:
        Distance(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::bigint id1, LAMMPS_NS::bigint id2, FILE *f_check);
        ~Distance() override;
        double compute_cv() override;
        void bias_force(double dVdcv) override;
        void summary(FILE* f) override;
        void get_dcvdx(double cv_value, double *dcvdx) override;
        void delta_x();
    };

    // Steinhardt local
    class Steinhardt : public CV {
    protected:
        LAMMPS_NS::bigint *atoms;
        bool pbc_x, pbc_y, pbc_z;
        double box_x, box_y, box_z;
        double dx, dy, dz;
    public:
        Steinhardt(LAMMPS_NS::LAMMPS *lmp, FILE *f_check):CV(lmp, f_check){}
        virtual ~Steinhardt() override;
        void summary(FILE* f) override = 0;
        virtual double compute_cv() = 0;
        virtual void bias_force(double dVdcv) override = 0;
        virtual void get_dcvdx(double cv_value, double *dcvdx) override = 0;
    };

    // Steinhardt local
    class STEIN_Q4 : public Steinhardt {
    private:
        int stein_l=4;
        FILE *file = nullptr;
        LAMMPS_NS::Error *error = nullptr;
        double cutoff_r;          // envioronment_cutoff radius
        int cutoff_Natoms;        // envioronment_cutoff natoms
        int d_block_size;         // use it to change the GPU set
        int GPU_number;
        int block_num;
        int neighbor_type = 0; 
        size_t N;
        double cv_value;
        double **stein_q = nullptr;
        int group_count, group_id, groupbit;
        LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
        // [nlist] : full neighborlist
        LAMMPS_NS::NeighList *nlist = nullptr;
        // lammps imformation
        LAMMPS_NS::Atom *atom = nullptr;
        int *numneigh = nullptr;            // ptr to get the list->numneigh
        int **firstneigh = nullptr;         // ptr to get the list->firstneigh
        // [mask] : list for lammps each group id, 1-D = [nlocal]
        //          e.g. when use "group test id 1 1000 5000", find its groupid by "test"
        //               then we can find atoms in this group by use "mask[i] & groupid"
        // [group_indices] : group atoms tagint, 1-D = [atoms in group and also in local]
        int *mask = nullptr;
        int *d_mask = nullptr;
        int *h_group_indices = nullptr;
        int *d_group_indices = nullptr;
        // [group_numneigh] : group neighbor, neighbors number for center atoms
        LAMMPS_NS::tagint *h_group_numneigh = nullptr;
        LAMMPS_NS::tagint *d_group_numneigh = nullptr;
        // [firstneigh_ptrs]: group neighbor, each center atoms * neighbors localtag
        int *h_firstneigh_ptrs = nullptr;
        int *d_firstneigh_ptrs = nullptr;
        // int Q_hybrid[4];
        // [h_x_flat] : list for lammps atoms, 3*nlocal
        double *h_x_flat = nullptr;
        double *d_x_flat = nullptr;
        // [h_stein_Ylm] : Ylm for each c_atoms
        double *h_stein_Ylm = nullptr;
        // double *Q_per_atoms_value = nullptr;
        // [group_dminneigh] = [ delta x, delta y, delta z, r squared] * c_atoms * cutoff_N ]
        double *group_dminneigh = nullptr;
        double *d_group_dminneigh = nullptr;
        // [dYlm_dr] = dcv/dx (complex add local)
        // [dcvdx] = dcv/dx
        double *d_stein_Ylm = nullptr;
        double *d_dYlm_dr = nullptr;
        double *h_dYlm_dr = nullptr;
        double *d_dcvdx = nullptr;
        double *h_dcvdx = nullptr;
        // device local
        // stein_ql in host is stored in steinq[i]
        // stein_qlm stored in 
        double *d_stein_ql = nullptr;
        double *d_stein_qlm = nullptr;
        double *h_stein_qlm = nullptr;
        // [neigh_in_cutoff_r] : how many neigh's r less than set
        // [neigh_both_in_r_N] : how many neighs satisfied r and N
        // [calculated_numneigh] : local tagint of neighs, both in r and N
        //                         default tagint is -1, 1-D = [c_atoms*cutoff_N]
        int *neigh_in_cutoff_r = nullptr;
        int *d_neigh_in_cutoff_r = nullptr;
        int *neigh_both_in_r_N = nullptr;
        int *d_neigh_both_in_r_N = nullptr;
        LAMMPS_NS::tagint *d_calculated_numneigh = nullptr;
        LAMMPS_NS::tagint *calculated_numneigh = nullptr;
    public:
        STEIN_Q4(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, int group_id, 
                                double cutoff_r, int cutoff_Natoms, int d_block_size);
        ~STEIN_Q4() override;
        double compute_cv() override;
        void bias_force(double dVdcv) override;
        void summary(FILE* f) override;
        void get_dcvdx(double cv_value, double *dcvdx) override;
        void envioronment();
        void steinhardt_param_calc(double *);
        void get_numneigh_full_pair_ABANDON_();
        // void fix_crystallizes_kernel_temp
        //     (int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
        //     int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
        //     int *d_firstneigh_ptrs, double *d_x_flat,
        //     double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N, int atomsnumber) ;
    };

    // // Steinhardt local
    // class Steinh_Q6 : public Steinhardt {
    // private:
    // public:
    //     Steinh_Q6(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::bigint *atoms, FILE *f_check);
    //     ~Steinh_Q6() override;
    //     double compute_cv() override;
    //     void bias_force(double dVdcv) override;
    //     void summary(FILE* f) override;
    //     void get_dcvdx(double cv_value, double *dcvdx) override;
    // };

    // // Steinhardt local
    // class Steinh_LQ4 : public Steinhardt {
    // private:
    // public:
    //     Steinh_LQ4(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::bigint *atoms, FILE *f_check);
    //     ~Steinh_LQ4() override;
    //     double compute_cv() override;
    //     void bias_force(double dVdcv) override;
    //     void summary(FILE* f) override;
    //     void get_dcvdx(double cv_value, double *dcvdx) override;
    // };

    // // Steinhardt local
    // class Steinh_LQ6 : public Steinhardt {
    // private:
    // public:
    //     Steinh_LQ6(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::bigint *atoms, FILE *f_check);
    //     ~Steinh_LQ6() override;
    //     double compute_cv() override;
    //     void bias_force(double dVdcv) override;
    //     void summary(FILE* f) override;
    //     void get_dcvdx(double cv_value, double *dcvdx) override;
    // };

    Steinhardt* create_steinhardt_cv(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                                int group_id, int Q_num, 
                                char *Q_type_str, double cutoff_r, int cutoff_Natoms, 
                                int d_block_size);
}