#include "fix_crystallize.h"
#include "lammps.h"
#include "pair.h"
#include "neigh_request.h"

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
        void compute_grad(double dVdcv) override;
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
        virtual void compute_grad(double dVdcv) override = 0;
        virtual void get_dcvdx(double cv_value, double *dcvdx) override = 0;
    };

    // Steinhardt local
    class STEIN_Q4 : public Steinhardt {
    private:
        FILE *file;
        LAMMPS_NS::Error *error;
        LAMMPS_NS::Atom *atom;
        int d_block_size;         // use it to change the GPU set
        int GPU_number;
        int block_num;
        int neighbor_type = 0; 
        size_t N;
        double **stein_q;
        int group_count, group_id, groupbit;
        // LAMMPS_NS::Pair *pair;         // ptr to pair style that uses neighbor history
        LAMMPS_NS::FixMetadynamics *Fixmetad;
        LAMMPS_NS::NeighList *nlist;    // this list init in init() and getby init_list(),
                            // which store the NeighConst::REQ_FULL
        int *numneigh;            // ptr to get the list->numneigh
        int **firstneigh;         // ptr to get the list->firstneigh
        double cutoff_r;          // envioronment_cutoff radius
        int cutoff_Natoms;        // envioronment_cutoff natoms
        int *neigh_in_cutoff_r;   // ptr alloc, how many neigh's r less than set
        int *neigh_both_in_r_N;   // ptr alloc, how many neighs satisfied r and N
        int *h_group_indices;
        // int Q_hybrid[4];          // use it to set 
        double *Q_per_atoms_value;
        double *group_dminneigh;
    public:
        STEIN_Q4(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, int group_id, 
                                double cutoff_r, int cutoff_Natoms, int d_block_size);
        ~STEIN_Q4() override;
        double compute_cv() override;
        void compute_grad(double dVdcv) override;
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
    //     void compute_grad(double dVdcv) override;
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
    //     void compute_grad(double dVdcv) override;
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
    //     void compute_grad(double dVdcv) override;
    //     void summary(FILE* f) override;
    //     void get_dcvdx(double cv_value, double *dcvdx) override;
    // };

    Steinhardt* create_steinhardt_cv(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                                int group_id, int Q_num, 
                                char *Q_type_str, double cutoff_r, int cutoff_Natoms, 
                                int d_block_size);
}