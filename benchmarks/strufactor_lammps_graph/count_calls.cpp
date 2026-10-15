// 仅用于 LD_PRELOAD 调用计数：共享环境 kernel 及插件发起的 MPI_SUM。
// 不参与正常构建，也不要在性能计时中启用。MPI 通过 PMPI 转发，CUDA 通过 RTLD_NEXT 转发。
#include <cuda_runtime.h>
#include <mpi.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static unsigned long env_launches, scalar_sum, scalar_count, packed_sum;
extern "C" cudaError_t cudaLaunchKernel(const void* f, dim3 grid, dim3 block, void** args, size_t shared, cudaStream_t stream) {
    using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    static auto real = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "cudaLaunchKernel"));
    Dl_info info{};
    if (dladdr(f, &info) && info.dli_sname && strstr(info.dli_sname, "get_environment_Strufactor")) ++env_launches;
    return real(f, grid, block, args, shared, stream);
}
extern "C" int MPI_Allreduce(const void* in, void* out, int n, MPI_Datatype type, MPI_Op op, MPI_Comm comm) {
    Dl_info info{};
    if (dladdr(__builtin_return_address(0), &info) && info.dli_fname &&
        (strstr(info.dli_fname, "fix_crystallize_plugin.so") || strstr(info.dli_fname, "before-shared-mean.so")) && op == MPI_SUM) {
        if (type == MPI_DOUBLE && n == 1) ++scalar_sum;
        if (type == MPI_LONG_LONG_INT && n == 1) ++scalar_count;
        if (type == MPI_DOUBLE && n == 2) ++packed_sum;
    }
    return PMPI_Allreduce(in, out, n, type, op, comm);
}
__attribute__((destructor)) static void report() {
    if (env_launches) fprintf(stderr,"PROBE rank=%s env=%lu sum1=%lu count1=%lu pair2=%lu\n",getenv("OMPI_COMM_WORLD_RANK"),env_launches,scalar_sum,scalar_count,packed_sum);
}
