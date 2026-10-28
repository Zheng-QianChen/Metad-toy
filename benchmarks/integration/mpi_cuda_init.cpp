#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
void check(cudaError_t status) { if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
int main(int argc,char**argv) {
    bool early=argc>1 && !std::strcmp(argv[1],"early-cuda");
    bool exchange=argc>1 && !std::strcmp(argv[1],"device-exchange");
    if(early)check(cudaFree(nullptr));
    MPI_Init(&argc,&argv);
    check(cudaFree(nullptr));
    if(exchange) {
        int *send,*receive,value=42,got=0;
        check(cudaMalloc(&send,sizeof(int)));check(cudaMalloc(&receive,sizeof(int)));
        check(cudaMemcpy(send,&value,sizeof(int),cudaMemcpyHostToDevice));
        MPI_Sendrecv(send,1,MPI_INT,0,123,receive,1,MPI_INT,0,123,MPI_COMM_SELF,MPI_STATUS_IGNORE);
        check(cudaMemcpy(&got,receive,sizeof(int),cudaMemcpyDeviceToHost));
        if(got!=value)MPI_Abort(MPI_COMM_WORLD,42);
        check(cudaFree(send));check(cudaFree(receive));
    }
    MPI_Finalize();
    std::puts("MPI/CUDA lifecycle completed; no LAMMPS or MetaDtoy loaded");
}
