#!/usr/bin/env python3
"""四项原型的统一构建与验证入口。

每项在独立子进程中执行，stdout/stderr 保存到 build 目录的 logs/。
数值指标、模型及图保存在 results/，suite.json 汇总命令、耗时和通过状态。
modules 可选择部分原型；arrays 仅配置本目录下的独立 CMake 项目。
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone

LAB=Path(__file__).resolve().parent
REPOSITORY=LAB.parent.parent


def tracked_hashes():
    """返回仓库中现存 Git 跟踪文件的 SHA256 映射，作为运行前后的内容基线。
    
    键为仓库相对路径，值为文件摘要；在没有 .git 的源码副本中返回 None。
    未跟踪的原型文件和 build 产物由各自的源码检查、运行日志另外管理。
    """
    if not (REPOSITORY/'.git').exists():return None
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=REPOSITORY).decode().split('\0')
    return {path:hashlib.sha256((REPOSITORY/path).read_bytes()).hexdigest() for path in paths if path and (REPOSITORY/path).is_file()}


def execute(name,command,logs,environment,records):
    """同步执行一条命令，保存日志并在 records 中追加运行记录。
    
    command 是参数列表，以避免经过 shell 再解释路径或特殊字符。
    成功时返回；失败时显示日志尾部并抛出异常，统一入口据此标记 suite 失败。
    """
    print(f'[{name}]',flush=True)
    start=time.perf_counter();log=logs/(name+'.log')
    with log.open('w') as handle:
        handle.write('Command: '+repr([str(argument) for argument in command])+'\n');handle.flush()
        completed=subprocess.run([str(argument) for argument in command],cwd=REPOSITORY,env=environment,stdout=handle,stderr=subprocess.STDOUT)
    records.append({'name':name,'returncode':completed.returncode,'seconds':time.perf_counter()-start,'log':str(log)})
    if completed.returncode:
        print(log.read_text(errors='replace')[-10000:],file=sys.stderr)
        raise RuntimeError(f'{name} failed; see {log}')


def main():
    """解析路径、模块和工具配置，依次运行各验证并保存最终汇总。
    
    Python 子进程使用当前 sys.executable，MPI 使用调用方指定的 launcher。
    python-deps 用于已有 pip --target 目录，build-dir 用于编译及运行产物。
    测试阶段前后比较跟踪文件摘要，保证这次运行没有改变已记录的项目内容。
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modules',nargs='+',choices=['arrays','sampling','ml','highdim'],default=['arrays','sampling','ml','highdim'])
    parser.add_argument('--build-dir',type=Path,default=REPOSITORY/'build/extension_lab')
    parser.add_argument('--python-deps',type=Path,help='Optional pip --target directory added to PYTHONPATH')
    parser.add_argument('--lammps-source',type=Path)
    parser.add_argument('--gpumd-source',type=Path)
    parser.add_argument('--cuda-root',type=Path,default=Path('/usr/local/cuda'))
    parser.add_argument('--cuda-arch',help='Kokkos architecture suffix, e.g. ADA89, AMPERE80')
    parser.add_argument('--cmake',default=shutil.which('cmake') or 'cmake')
    parser.add_argument('--mpi-cxx',default=shutil.which('mpicxx') or 'mpicxx')
    parser.add_argument('--mpiexec',default=shutil.which('mpirun') or 'mpirun')
    parser.add_argument('--mpi-extra',action='append',default=[],help='One additional launcher argument; e.g. --mpi-extra=--oversubscribe')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--sanitize',action='store_true')
    args=parser.parse_args()
    # 验证器使用 assert 检查指标；Python -O 会去掉这些检查，因此入口直接拒绝。
    if sys.flags.optimize:parser.error('Run validations without Python -O')
    if 'arrays' in args.modules:
        if not args.lammps_source or not args.gpumd_source:parser.error('arrays requires --lammps-source and --gpumd-source')
        args.lammps_source=args.lammps_source.resolve();args.gpumd_source=args.gpumd_source.resolve()
    base=args.build_dir.resolve();results=base/'results';logs=base/'logs'
    results.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    # 为子进程创建环境副本，依赖目录与线程数设置只作用于本次验证链。
    environment=os.environ.copy()
    if args.python_deps:
        environment['PYTHONPATH']=str(args.python_deps.resolve())+os.pathsep+environment.get('PYTHONPATH','')
    environment['PATH']=str(args.cuda_root/'bin')+os.pathsep+environment.get('PATH','')
    environment.setdefault('TF_CPP_MIN_LOG_LEVEL','2');environment.setdefault('TF_NUM_INTRAOP_THREADS','2');environment.setdefault('TF_NUM_INTEROP_THREADS','2');environment.setdefault('OMP_NUM_THREADS','2')
    # 本原型 MPI 交换的是 Host 标量，使用标准传输即可。
    environment.setdefault('OMPI_MCA_btl','self,tcp')
    records=[];before=tracked_hashes()
    # 开始时先标记 running，结束时在 finally 写回 pass/failed，便于回查中途失败。
    summary={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'modules':args.modules,'commands':records}
    (results/'suite.json').write_text(json.dumps(summary,indent=2))
    def run(name,command):execute(name,command,logs,environment,records)
    def python_test(name,path):run(name,[sys.executable,LAB/path,'--output',results])
    try:
        if 'arrays' in args.modules:
            configuration=[args.cmake,'-S',LAB/'external_arrays','-B',base/'arrays',f'-DCMAKE_CXX_COMPILER={args.lammps_source}/lib/kokkos/bin/nvcc_wrapper',f'-DLAMMPS_SOURCE={args.lammps_source}',f'-DGPUMD_SOURCE={args.gpumd_source}',f'-DCUDAToolkit_ROOT={args.cuda_root}',f'-DMPI_CXX_COMPILER={args.mpi_cxx}','-DCMAKE_BUILD_TYPE=Release']
            if args.cuda_arch:configuration.append(f'-DKokkos_ARCH_{args.cuda_arch}=ON')
            run('arrays_configure',configuration)
            run('arrays_build',[args.cmake,'--build',base/'arrays','-j',args.jobs])
            array_results=[]
            # 同一个测试源码编译成两个目标，分别覆盖 LAMMPS 的两种坐标布局。
            for target in ('array_bridge','array_bridge_layoutleft'):
                run(target,[base/'arrays'/target])
                lines=(logs/(target+'.log')).read_text().splitlines()
                result=json.loads(next(line for line in lines if line.startswith('{')))
                result['layout_output']=next(line for line in lines if line.startswith('Kokkos coordinate'))
                array_results.append(result)
                # 显存检查发现问题时返回 99，让统一入口将该项视为失败。
                if args.sanitize:run(target+'_memcheck',[args.cuda_root/'bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',base/'arrays'/target])
            (results/'external_arrays.json').write_text(json.dumps({'status':'pass','cases':array_results,'memcheck':args.sanitize},indent=2))
        if 'sampling' in args.modules:
            python_test('sampling','sampling/validate.py')
            # 总进程数固定为 4；分别检验 4×1 rank 和 2×2 rank 的副本组织。
            for ranks in (1,2):
                run(f'replica_{4//ranks}x{ranks}',[args.mpiexec,*args.mpi_extra,'-np','4',sys.executable,LAB/'sampling/replica_exchange.py','--output',results,'--ranks-per-replica',ranks])
        if 'ml' in args.modules:python_test('ml_models','ml_models/validate.py')
        if 'highdim' in args.modules:python_test('highdim','highdim/validate.py')
        after=tracked_hashes()
        if before!=after:raise RuntimeError('Tracked project files changed during validation')
        summary['unchanged_tracked_files']=None if before is None else len(before)
        summary['status']='pass'
    except Exception as error:
        summary['status']='failed';summary['error']=str(error)
        raise
    finally:
        summary['finished_utc']=datetime.now(timezone.utc).isoformat()
        (results/'suite.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'PASS: {results / "suite.json"}',flush=True)

if __name__=='__main__':main()
