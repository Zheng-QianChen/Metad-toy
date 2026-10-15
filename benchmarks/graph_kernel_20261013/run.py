#!/usr/bin/env python3
import argparse, hashlib, json, os, shlex, shutil, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',default=str(HERE/'results'))
parser.add_argument('--build-dir',default=str(ROOT/'build/graph'))
parser.add_argument('--run-only',action='store_true')
parser.add_argument('--focus',action='store_true',help='21 rounds, pair only, N=1048576 and 4194304')
args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
BUILD=Path(args.build_dir).resolve()
NVCC=os.environ.get('NVCC','/usr/local/cuda/bin/nvcc')
CMAKE=os.environ.get('CMAKE',shutil.which('cmake') or 'cmake')
LAMMPS=os.environ.get('LAMMPS_SOURCE_DIR','/home/zqc/apps/lammps/lammps-stable_29Aug2024/src')
commands=[]
def run(cmd,log):
    print('+',shlex.join(cmd),flush=True);commands.append(cmd)
    with open(out/log,'w') as handle:
        result=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        print((out/log).read_text());raise SystemExit(result.returncode)
if not args.run_only:
    run([CMAKE,'-S',str(ROOT),'-B',str(BUILD),'-DZQC_GRAPH_STANDALONE=ON','-DLAMMPS_SOURCE_DIR='+LAMMPS,'-DCMAKE_CUDA_COMPILER='+NVCC,'-DCMAKE_CUDA_ARCHITECTURES=89'],'project_configure.log')
    run([CMAKE,'--build',str(BUILD),'--target','zqc_graph_smoke','zqc_graph_emit','-j2'],'project_build.log')
    run([str(BUILD/'benchmarks/graph/zqc_graph_emit'),str(out)],'emit_run.log')
    for case in ['mean_exp','two_dims']:
        shutil.copy2(HERE/(case+'.manual.cu'),out/(case+'.manual.cu'))
        run([NVCC,'-std=c++17','-O3','-arch=sm_89','--cubin',str(HERE/(case+'.manual.cu')),'-o',str(out/(case+'.manual_nvcc.cubin'))],case+'.nvcc.log')
    run([NVCC,'-std=c++17','-O3','-arch=sm_89',str(HERE/'benchmark.cu'),'-L/usr/lib/wsl/lib','-lcuda','-lnvrtc','-o',str(out/'benchmark')],'benchmark_build.log')
    metadata={
        'commands':commands,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'include/utils/zqc_graph.cuh',ROOT/'src/utils/zqc_graph.cu',ROOT/'src/utils/zqc_graph_nodes.cu',ROOT/'src/utils/zqc_graph_parser.cu',ROOT/'src/utils/zqc_graph_execution.cu',ROOT/'include/utils/zqc_graph_device.cuh',ROOT/'include/utils/zqc_gpu_buffer.cuh',HERE/'benchmark.cu',HERE/'mean_exp.manual.cu',HERE/'two_dims.manual.cu']},
        'gpu':subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip(),
        'nvcc':subprocess.check_output([NVCC,'--version'],text=True),
        'mode':'focus' if args.focus else 'full',
        'block_size':256,'rounds':21 if args.focus else 9,'kernels_per_batch':64,'target_sample_ms':5,
        'sizes':[1048576,4194304] if args.focus else [3,1024,16384,65536,262144,1048576,4194304],
        'phases':['pair'] if args.focus else ['forward','backward','pair'],
        'notes':'FP64; identical Driver API launch and buffers; no restrict or fast-math; warm repeated buffers. CUDA Graph replay batches amortize CPU submission gaps. Pair means one forward + one backward. Mean reduction excluded for all variants. Desktop WSL GPU; no clock locking.'}
    shutil.copy2(HERE/'benchmark.cu',out/'benchmark_used.cu')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
print('Benchmark running; progress is written to '+str(out/'run.log'),flush=True)
with open(out/'run.log','w') as handle:
    result=subprocess.run([str(out/'benchmark'),str(out)]+(['focus'] if args.focus else []),stdout=handle,stderr=subprocess.STDOUT)
print((out/'run.log').read_text(),flush=True)
if result.returncode == 0:
    cuobjdump=str(Path(NVCC).with_name('cuobjdump'))
    for case in ['mean_exp','two_dims']:
        for variant in ['graph_nvrtc','manual_nvrtc','manual_nvcc']:
            run([cuobjdump,'--dump-sass',str(out/(case+'.'+variant+'.cubin'))],case+'.'+variant+'.sass')
raise SystemExit(result.returncode)
