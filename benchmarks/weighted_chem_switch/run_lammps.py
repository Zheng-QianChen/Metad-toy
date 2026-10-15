#!/usr/bin/env python3
"""比较分离/内联 fermi 的完整 LAMMPS timestep，编译与预热独立于计时。"""
import argparse
import json
import os
import random
import re
import statistics
import subprocess
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = spec_from_file_location("css_fixture",ROOT/"benchmarks/css/run.py")
fixture = module_from_spec(_spec)
_spec.loader.exec_module(fixture)


def initial(path,cells):
    atoms=[]
    for x in range(cells):
        for y in range(cells):
            for z in range(cells):
                for basis,(dx,dy,dz) in enumerate([(0,0,0),(.5,.5,0),(.5,0,.5),(0,.5,.5)]):
                    atom_type=2 if basis==0 else (3 if basis==3 and (x+y+z)%2==0 else 1)
                    atoms.append(f"{len(atoms)+1} {atom_type} {4.05*(x+dx):.17g} {4.05*(y+dy):.17g} {4.05*(z+dz):.17g}")
    length=4.05*cells
    path.write_text(f"Weighted chemical switching benchmark\n\n{len(atoms)} atoms\n3 atom types\n\n0 {length} xlo xhi\n0 {length} ylo yhi\n0 {length} zlo zhi\n\nMasses\n\n1 30\n2 30\n3 30\n\nAtoms # atomic\n\n"+"\n".join(atoms)+"\n")


def input_text(plugin,variant,data,cells,steps,warmup,validate):
    text=fixture.input_text(plugin,"GRAPH",cells,steps,warmup,validate,data)
    expression="AVE(fermi(w,-4,0.4))" if variant=="separate" else "AVE(w)"
    declarations=f"""CAL NAME W WEIGHT_CHEM all cutoff_r 3.05 d_block_size 128 Chem_map (1,1.0) (2,-0.7) (3,0.35) &
    SYMBOL w W.RAW &
    CV_dim 1 DIM 1 0 1 500 "{expression}" &
    """
    start=text.index("CAL NAME S1 STRU_FACTOR")
    end=text.index("METAD_RESTART",start)
    text=text[:start]+declarations+text[end:]
    text=text.replace("thermo 100000","compute bias_pressure all pressure NULL virial\nthermo 100000")
    text=text.replace("step atoms f_m[1] f_m[2] f_m","step atoms f_m[1] f_m "+" ".join(f"c_bias_pressure[{component}]" for component in range(1,7)))
    return text


def execute(directory,text,rank_count):
    fixture.execute(directory,text,rank_count)
    screen=(directory/"screen.txt").read_text()
    loops=re.findall(r"Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps with (\d+) atoms",screen)
    seconds,steps,atoms=loops[-1]
    compile_times=re.findall(r"NVRTC ready: compile_ms=([0-9.]+)",screen)
    assert len(compile_times)==1,compile_times
    return dict(us_per_step=1e6*float(seconds)/int(steps),loop_seconds=float(seconds),steps=int(steps),
                atoms=int(atoms),nvrtc_ms=float(compile_times[0]),directory=str(directory))


def validate(first,second):
    a=fixture.last_dump(first/"atom_state.dump")
    b=fixture.last_dump(second/"atom_state.dump")
    force_error=max(abs(row[c]-b[atom][c]) for atom,row in a.items() for c in range(3,6))
    position_error=max(abs(row[c]-b[atom][c]) for atom,row in a.items() for c in range(3))
    magnitude=max(abs(row[c]) for row in a.values() for c in range(3,6))
    observables_a=fixture.thermo(first/"screen.txt")
    observables_b=fixture.thermo(second/"screen.txt")
    observable_error=max(abs(x-y) for x,y in zip(observables_a[:2],observables_b[:2]))
    pressure_error=max(abs(x-y) for x,y in zip(observables_a[2:],observables_b[2:]))
    assert magnitude>1e-10 and force_error<1e-10 and position_error<1e-10 and observable_error<1e-10 and pressure_error<1e-9,(magnitude,force_error,pressure_error)
    return dict(force_error=force_error,position_error=position_error,cv_energy_error=observable_error,
                pressure_error=pressure_error,max_force=magnitude)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cells",type=int,default=10)
    parser.add_argument("--steps",type=int,default=20000)
    parser.add_argument("--warmup",type=int,default=1000)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--np",type=int,nargs="+",default=[1,2])
    args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    data=args.output/"initial.data";initial(data,args.cells)
    plugins={"separate":ROOT/"build/graph-lammps/fix_crystallize_plugin.so","fused":args.prepared.resolve()/"fused/fix_crystallize_plugin.so"}
    validation=[]
    for rank_count in args.np:
        paths={}
        for variant in plugins:
            paths[variant]=args.output/f"validate_{variant}_np{rank_count}"
            execute(paths[variant],input_text(plugins[variant],variant,data,args.cells,1,0,True),rank_count)
        result=dict(ranks=rank_count,**validate(paths["separate"],paths["fused"]))
        validation.append(result);print("VALIDATION "+json.dumps(result),flush=True)
    (args.output/"validation.json").write_text(json.dumps(validation,indent=2))
    results=[];randomizer=random.Random(20261015)
    for rank_count in args.np:
        for repeat in range(args.repeats):
            variants=list(plugins);randomizer.shuffle(variants)
            for variant in variants:
                directory=args.output/f"{variant}_np{rank_count}_r{repeat}"
                result=execute(directory,input_text(plugins[variant],variant,data,args.cells,args.steps,args.warmup,False),rank_count)
                result.update(variant=variant,ranks=rank_count,repeat=repeat)
                results.append(result);print(json.dumps(result),flush=True)
                (args.output/"results.json").write_text(json.dumps(results,indent=2))
    summary=[]
    for rank_count in args.np:
        samples={variant:[row["us_per_step"] for row in results if row["variant"]==variant and row["ranks"]==rank_count] for variant in plugins}
        medians={variant:statistics.median(values) for variant,values in samples.items()}
        paired=[next(row["us_per_step"] for row in results if row["ranks"]==rank_count and row["repeat"]==repeat and row["variant"]=="separate") /
                next(row["us_per_step"] for row in results if row["ranks"]==rank_count and row["repeat"]==repeat and row["variant"]=="fused") for repeat in range(args.repeats)]
        summary.append(dict(ranks=rank_count,median_us=medians,fused_speedup=medians["separate"]/medians["fused"],paired_speedups=paired,
                            sample_ranges={variant:[min(values),max(values)] for variant,values in samples.items()}))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print("LAMMPS_SWITCH_BENCHMARK_PASS "+json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
