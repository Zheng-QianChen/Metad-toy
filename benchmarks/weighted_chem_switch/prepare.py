#!/usr/bin/env python3
"""从当前生产 kernel 生成独立融合对照；生产目录保持不变。"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def source_hashes():
    paths = [path for directory in ("src", "include", "cmake") for path in (ROOT/directory).rglob("*") if path.is_file()]
    paths.append(ROOT/"CMakeLists.txt")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new, 1)


def kernel_definition(source, name):
    declaration = source.rindex("__global__ void " + name)
    start = source.rindex("template<", 0, declaration)
    opening = source.index("{", declaration)
    depth = 1
    end = opening+1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def generate_fused_kernels():
    source = (ROOT/"src/cvs/CVs_Strufactor.cu").read_text()
    forward = kernel_definition(source, "forward_kernel").replace("forward_kernel", "fused_forward_kernel", 1)
    forward = replace_once(forward, "double* values) {", "double* raw_values, double* values) {")
    forward = replace_once(forward, "values[view.d_group_indices[k]] = value;",
                           "raw_values[view.d_group_indices[k]] = value;\n    values[view.d_group_indices[k]] = zqc_graph_fermi(value,-4.0,0.4);")
    backward = kernel_definition(source, "backward_kernel").replace("backward_kernel", "fused_backward_kernel", 1)
    backward = replace_once(backward, "const double* adjoint, double* force, double* virial)",
                             "const double* raw_values, const double* adjoint, double* force, double* virial)")
    backward = replace_once(backward, "const double seed = adjoint[k];",
                             "const double seed = adjoint[k]*zqc_graph_fermi_df(raw_values[view.d_group_indices[k]],-4.0,0.4);")
    return '#include "zqc_graph_device.cuh"\nnamespace MetaD_zqc {\nnamespace {\n' + forward + "\n" + backward + "\n}\n" + r"""
void fused_weighted_forward(const StruFactorData& data, double cutoff_rsq,
    const int* atom_types, const double* type_weights, double* raw_values, double* values,
    cudaStream_t stream, int block_size) {
    const auto view=kernel_view(data);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    fused_forward_kernel<true,false><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,0.0,cutoff_rsq,atom_types,type_weights,raw_values,values);
    check_launch();
}
void fused_weighted_backward(const StruFactorData& data, double cutoff_rsq,
    const int* atom_types, const double* type_weights, const double* raw_values,
    const double* seeds, double* forces, double* virials, cudaStream_t stream, int block_size) {
    const auto view=kernel_view(data);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    fused_backward_kernel<true,false><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,0.0,cutoff_rsq,atom_types,type_weights,raw_values,seeds,forces,virials);
    check_launch();
}
} // namespace MetaD_zqc
"""


def command(arguments, log):
    print("RUN " + " ".join(map(str, arguments)), flush=True)
    environment = os.environ.copy()
    environment["OMPI_CXX"] = "/usr/bin/g++"
    with log.open("w") as handle:
        result = subprocess.run(list(map(str, arguments)), stdout=handle, stderr=subprocess.STDOUT, env=environment)
    if result.returncode:
        raise RuntimeError(log.read_text()[-10000:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"build/weighted-chem-switch")
    parser.add_argument("--resume",action="store_true",help="Continue compiling an already prepared benchmark copy")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    (output/"fused_kernels.cuh").write_text(generate_fused_kernels())
    # 只复制构建所需文件，临时插件在 build 中增加 raw W 缓冲并内联同一个 fermi/df。
    source = output/"source"
    if not arguments.resume:
        source.mkdir(exist_ok=False)
        for directory in ("src", "include", "cmake", "third_party"):
            shutil.copytree(ROOT/directory, source/directory)
        shutil.copy2(ROOT/"CMakeLists.txt", source/"CMakeLists.txt")
        definitions = """
    void fused_weighted_forward(const StruFactorData&, double, const int*, const double*, double*, double*, cudaStream_t, int);
    void fused_weighted_backward(const StruFactorData&, double, const int*, const double*, const double*, const double*, double*, double*, cudaStream_t, int);
    """
        header = source/"include/CVsets/CV_Weighted_Chemical_Pair_Order.h"
        text = header.read_text().replace("namespace MetaD_zqc {", "namespace MetaD_zqc {\n"+definitions, 1)
        text = text.replace("private:\n", "private:\n    GpuBuffer<double> d_unswitched_values; // 实验：反向复用前向的原始 W。\n", 1)
        header.write_text(text)
        implementation = source/"src/cvs/CV_Weighted_Chemical_Pair_Order.cu"
        text = implementation.read_text()
        text = replace_once(text, "my_chem_env(environment) {}", 'my_chem_env(environment) { register_buffer(d_unswitched_values,"bench_raw_W"); }')
        text = replace_once(text, "    weighted_chem_forward(my_env->data", "    d_unswitched_values.grow_to(lmp->atom->nlocal+lmp->atom->nghost,__FILE__,__LINE__);\n    fused_weighted_forward(my_env->data")
        text = replace_once(text, "                          d_stru_factor.ptr,0,d_block_size);", "                          d_unswitched_values.ptr,d_stru_factor.ptr,0,d_block_size);")
        text = replace_once(text, "    weighted_chem_backward(my_env->data", "    fused_weighted_backward(my_env->data")
        text = replace_once(text, "                           center_adjoint,atom_forces,atom_virials,0,d_block_size);", "                           d_unswitched_values.ptr,center_adjoint,atom_forces,atom_virials,0,d_block_size);")
        implementation.write_text(text)
        implementation = source/"src/cvs/CVs_Strufactor.cu"
        implementation.write_text(implementation.read_text()+'\n#include "'+str(output/"fused_kernels.cuh")+'"\n')
    cmake = Path("/home/zqc/apps/cmake/3.31.8/bin/cmake")
    lammps = Path("/home/zqc/apps/lammps/lammps-stable_29Aug2024/src")
    mpi = Path("/home/zqc/apps/deepmd")
    command([cmake,"-S",source,"-B",output/"fused", "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_DEBUG=OFF", "-DUSE_ML_CVS=OFF",
             "-DCMAKE_CUDA_ARCHITECTURES=89", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc", "-DCLEAN_HOST_COMPILER=/usr/bin/g++", "-DLAMMPS_SOURCE_DIR="+str(lammps),
             "-DLAMMPS_LIB_DIR="+str(mpi/"lib"), "-DMPI_HOME="+str(mpi)], output/"configure.log")
    command([cmake,"--build",output/"fused","-j4"],output/"plugin-build.log")
    command(["/usr/local/cuda/bin/nvcc","-O3","-std=c++17","-arch=sm_89","-lineinfo","-DFMT_HEADER_ONLY", "-ccbin","/usr/bin/g++",
             "-I"+str(output),"-I"+str(ROOT/"include/utils"),"-I"+str(ROOT/"include/CVsets"),"-I"+str(lammps),"-I"+str(mpi/"include"),
             HERE/"benchmark.cu",ROOT/"build/graph/benchmarks/graph/libzqc_graph_runtime.a",ROOT/"build/graph/libzqc_graph_embedded.a",
             "-L"+str(mpi/"lib"),"-L/usr/lib/wsl/lib","-Xlinker","-rpath","-Xlinker",str(mpi/"lib"),
             "-lmpi","-lcuda","-lnvrtc","-o",output/"benchmark"],output/"micro-build.log")
    assert source_hashes() == before, "production changed during preparation"
    (output/"metadata.json").write_text(json.dumps(dict(production_sha256=before,production_unchanged=True,
        fused_header_sha256=hashlib.sha256((output/"fused_kernels.cuh").read_bytes()).hexdigest()),indent=2))
    print("SWITCH_BENCHMARK_PREPARED",flush=True)


if __name__ == "__main__":
    main()
