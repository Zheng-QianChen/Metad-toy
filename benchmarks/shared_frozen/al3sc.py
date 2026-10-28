#!/usr/bin/env python3
"""Al3Sc 冻结共享偏势验证；在独立目录中运行，所有输入由参数明确给出。

沿用 LocalQ6 × CSS × fermi(LocalQ6 × CSS) 描述符和 Al-Sc MEAM。
小体系检查偏置力/六分量应变导数；--restart 可检查已有 6912 原子初态。
结果同时保存物理 pair 力参考、偏势能、逐原子描述符和跨后端可比较的状态。
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import numpy as np
from mpi4py import MPI
from lammps import lammps

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plugin",type=Path,required=True)
parser.add_argument("--assets",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--lammps-args",default="")
parser.add_argument("--weave-plugin",type=Path,help="Load matching Weave communicator before creating the box")
parser.add_argument("--raw-product",action="store_true",help="Use LocalQL6*CSS without an outer switch")
parser.add_argument("--aggregation",choices=['MEAN','SUM'],default='MEAN')
parser.add_argument("--restart",type=Path)
parser.add_argument("--single",action="store_true",help="Only atom 1 contributes, exercising empty local families")
parser.add_argument("--skip-fd",action="store_true")
parser.add_argument("--steps",type=int,default=30)
args=parser.parse_args()
comm=MPI.COMM_WORLD
plugin=args.plugin.resolve();assets=args.assets.resolve();output=args.output.resolve()
restart=args.restart.resolve() if args.restart else None
if comm.rank==0: output.mkdir(parents=True,exist_ok=False)
comm.Barrier()
os.chdir(output)
table=output/"shared.table"
if comm.rank==0:
    # 三次 Hermite 能精确再现此二次势，直接均值提供独立能量参考。
    q=np.linspace(-.2,1.2,141)
    table.write_text("# METAD_TOY_SHARED_FROZEN 1\n"+"".join(
        f"{value:.17g} {.4*(value-.35)**2:.17g} {.8*(value-.35):.17g}\n" for value in q))
    if not restart:
        lattice=4.15
        xyz=np.array([(i+dx,j+dy,k+dz) for i in range(3) for j in range(3) for k in range(3)
                      for dx,dy,dz in ((0,0,0),(.5,.5,0),(.5,0,.5),(0,.5,.5))])*lattice+.19
        ids=np.arange(1,len(xyz)+1)
        xyz+=np.column_stack([.07*np.sin(ids*.73),.06*np.cos(ids*.53),.05*np.sin(ids*.31)])
        (output/"initial.data").write_text("Al3Sc L12 perturbed validation\n\n108 atoms\n2 atom types\n\n"+
            "".join(f"0 {3*lattice} {axis}lo {axis}hi\n" for axis in "xyz")+
            "\nMasses\n\n1 26.9815385\n2 44.955908\n\nAtoms # atomic\n\n"+
            "".join(f"{i} {2 if (i-1)%4==0 else 1} {x:.17g} {y:.17g} {z:.17g}\n"
                    for i,(x,y,z) in zip(ids,xyz)))
comm.Barrier()
weave_setup=(f"plugin load {args.weave_plugin.resolve()}\nweave/batch_comm\n" if args.weave_plugin else "")
input_configuration=weave_setup+f"""units metal
atom_style atomic
newton on
atom_modify map array sort 5 0.0
boundary p p p
{"read_restart "+str(restart) if restart else "read_data initial.data"}
mass 1 26.9815385
mass 2 44.955908
pair_style meam
pair_coeff * * {assets/"Al-Sc.library.meam"} Al Sc {assets/"Al-Sc.meam"} Al Sc
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
comm_modify cutoff 12.0
group Sc type 2
group family {"id 1" if args.single else "type 2"}
plugin load {plugin}
thermo 100
thermo_style custom step pe
thermo_modify norm no format float %.17g
run 0
"""
descriptor="q*c" if args.raw_product else "q*c*fermi(q*c,0.35,15)"
bias_configuration=f"""fix bias all metad EXECUTOR GRAPH SHARED_FROZEN {table} family SHARED_AGGREGATION {args.aggregation} PACE 0 &
 CAL NAME SW_r6 SW_FUNC FERMI r_0 3.50 alpha -20 &
 CAL NAME SW_css SW_FUNC FERMI r_0 3.30 alpha -20 &
 CAL NAME LQ6 STEINH L 6 Sc cutoff_eps_r 1e-12 SW_FUNC_r SW_r6 &
 CAL NAME JOINT CSS Sc CHANNEL 2 SW_css 0 0.45 cutoff_eps 1e-12 &
 SYMBOL q LQ6.RAW SYMBOL c JOINT.RAW &
 CV_dim 1 DIM_EXPR 1 "{descriptor}"
compute pa all metad/atom bias LQ6.stein_q JOINT.css
compute bp all pressure NULL fix
compute stress all stress/atom NULL fix
compute stresssum all reduce sum c_stress[1] c_stress[2] c_stress[3] c_stress[4] c_stress[5] c_stress[6]
thermo_style custom step temp pe etotal f_bias f_bias[1] c_bp[1] c_bp[2] c_bp[3] c_bp[4] c_bp[5] c_bp[6] c_stresssum[1] c_stresssum[2] c_stresssum[3] c_stresssum[4] c_stresssum[5] c_stresssum[6]
run 0
"""
if comm.rank==0: (output/"in.lammps").write_text(input_configuration+bias_configuration)
simulation=lammps(cmdargs=["-screen","none","-log","log.lammps",*shlex.split(args.lammps_args)],comm=comm)

def gather(extra=False):
    count=simulation.extract_setting("nlocal")
    ids=simulation.numpy.extract_atom("id")[:count].copy()
    columns=[ids,simulation.numpy.extract_atom("type")[:count].copy(),
             simulation.numpy.extract_atom("x")[:count].copy(),simulation.numpy.extract_atom("f")[:count].copy()]
    if extra: columns.append(simulation.numpy.extract_compute("pa",1,2)[:count].copy())
    rows=np.concatenate(comm.allgather(np.column_stack(columns)))
    return rows[np.argsort(rows[:,0])]

def state():
    rows=gather(True)
    selected=rows[:,0]==1 if args.single else rows[:,1]==2
    product=rows[selected,8]*rows[selected,9]
    coordinate=product if args.raw_product else product/(1+np.exp(-15*(product-.35)))
    energy=float(simulation.extract_fix("bias",0,0))
    mean=float(simulation.extract_fix("bias",0,1,nrow=0))
    expected=float((np.mean if args.aggregation=='MEAN' else np.sum)(.4*(coordinate-.35)**2))
    if not np.isclose(energy,expected,rtol=1e-11,atol=1e-13): raise AssertionError(("energy reference",energy,expected))
    if not np.isclose(mean,np.mean(coordinate),rtol=1e-11,atol=1e-13): raise AssertionError("coordinate reference")
    # units metal: pressure [bar] * volume [A^3] / nktv2p = virial [eV].
    conversion=simulation.extract_global("nktv2p")
    pressure=simulation.numpy.extract_compute("bp",0,1).copy()
    stress=simulation.numpy.extract_compute("stresssum",0,1).copy()
    if not np.allclose(stress,-pressure*simulation.get_thermo("vol"),rtol=2e-10,atol=1e-9):
        raise AssertionError("per-atom virial sum")
    virial=pressure*simulation.get_thermo("vol")/conversion
    lo,hi,xy,yz,xz,_,_=simulation.extract_box()
    lengths=np.array(hi)-lo
    cell=[[lengths[0],xy,xz],[0,lengths[1],yz],[0,0,lengths[2]]]
    return dict(rows=rows.tolist(),energy=energy,mean=mean,virial=virial.tolist(),cell=cell,
                step=int(simulation.get_thermo("step")),total_energy=float(simulation.get_thermo("etotal")),
                family_count=int(selected.sum()),reference_error=abs(energy-expected),
                owned_counts=comm.allgather(simulation.extract_setting("nlocal")))

def place(coordinates):
    buffer=np.ascontiguousarray(coordinates,dtype=np.float64)
    simulation.scatter_atoms("x",1,3,buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    simulation.command("run 0")
    return float(simulation.extract_fix("bias",0,0))

try:
    simulation.commands_string(input_configuration)
    if not simulation.extract_global("triclinic"): simulation.command("change_box all triclinic")
    pair_rows=gather()
    simulation.commands_string(bias_configuration)
    baseline=state()
    coordinates=np.array(baseline["rows"])[:,2:5].copy()
    bias_forces=np.array(baseline["rows"])[:,5:8]-pair_rows[:,5:8]
    if np.max(np.abs(bias_forces.sum(axis=0)))>2e-10: raise AssertionError("net bias force")
    details=[]
    if not args.skip_fd:
        if restart: raise ValueError("Use --skip-fd for production restart; finite difference uses the small fixture")
        selected=[0,1,4,5,17,36,60,107]
        for step in (1e-5,3e-6):
            force_errors=[];force_scaled=[];virial_errors=[];virial_scaled=[]
            for atom in selected:
                for axis in range(3):
                    plus=coordinates.copy();minus=coordinates.copy()
                    plus[atom,axis]+=step;minus[atom,axis]-=step
                    numerical=-(place(plus)-place(minus))/(2*step)
                    analytic=bias_forces[atom,axis]
                    error=abs(numerical-analytic)
                    force_errors.append(error)
                    force_scaled.append(error/(2e-9+3e-5*max(abs(numerical),abs(analytic))))
            place(coordinates)
            for component,(row,column) in enumerate(((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))):
                energies=[]
                for sign in (1,-1):
                    matrix=np.eye(3);matrix[row,column]+=sign*step;cell=12.45*matrix
                    simulation.command(f"change_box all x final 0 {cell[0,0]:.17g} y final 0 {cell[1,1]:.17g} z final 0 {cell[2,2]:.17g} xy final {cell[0,1]:.17g} xz final {cell[0,2]:.17g} yz final {cell[1,2]:.17g} units box")
                    energies.append(place(coordinates@matrix.T))
                numerical=-(energies[0]-energies[1])/(2*step)
                analytic=baseline["virial"][component]
                error=abs(numerical-analytic)
                virial_errors.append(error)
                virial_scaled.append(error/(2e-8+3e-5*max(abs(numerical),abs(analytic))))
            simulation.command("change_box all x final 0 12.45 y final 0 12.45 z final 0 12.45 xy final 0 xz final 0 yz final 0 units box")
            place(coordinates)
            details.append(dict(step=step,max_force_error=max(force_errors),max_force_scaled=max(force_scaled),
                                max_virial_error=max(virial_errors),max_virial_scaled=max(virial_scaled)))
        if any(max(row["max_force_scaled"],row["max_virial_scaled"])>1 for row in details):
            raise AssertionError(details)
        recovered=state()
        if not np.allclose(recovered["rows"],baseline["rows"],rtol=0,atol=2e-10):
            raise AssertionError("coordinate/force restoration")
    # 固定表下作短 NVE 验证；本阶段不宣称完成 FES 采样。
    simulation.commands_string("timestep 0.0005\nfix integrate all nve\nthermo 10\n")
    if not restart: simulation.command("velocity all create 100 872635 mom yes rot no dist gaussian")
    frames=[state()]
    remaining=args.steps
    while remaining:
        count=min(10,remaining);simulation.command(f"run {count}")
        frames.append(state());remaining-=count
    before=state()
    simulation.command("write_restart restart.final")
    # 写 restart 可能重新排序；先完成一次求值，再读取依赖 localid 的 CV 输出。
    simulation.command("run 0")
    after=state()
    before_rows=np.array(before["rows"]);after_rows=np.array(after["rows"])
    delta=np.abs(before_rows-after_rows)
    if comm.rank==0:
        (output/"repeat_check.json").write_text(json.dumps(dict(
            max_by_column=delta.max(axis=0).tolist(),energy_difference=after["energy"]-before["energy"],
            virial_difference=(np.array(after["virial"])-before["virial"]).tolist(),
            largest_rows=np.argsort(delta.max(axis=1))[-5:].tolist(),
            before=before_rows[np.argsort(delta.max(axis=1))[-5:]].tolist(),
            after=after_rows[np.argsort(delta.max(axis=1))[-5:]].tolist()),indent=2))
    # 周期重映射可相差整数盒矢量；力、类型和 CV 仍逐项使用原来的严格容差。
    cell=np.array(after["cell"])
    fraction=(after_rows[:,2:5]-before_rows[:,2:5])@np.linalg.inv(cell).T
    equivalent=(fraction-np.rint(fraction))@cell.T
    checked=after_rows-before_rows;checked[:,2:5]=equivalent
    if np.max(np.abs(checked))>2e-10: raise AssertionError("repeated evaluation")
    if abs(after["energy"]-before["energy"])>1e-12 or np.max(np.abs(np.array(after["virial"])-before["virial"]))>1e-11:
        raise AssertionError("restart write changed bias energy/virial")
    # 重新装配同一个 selection 的成员，检查旧的 owned 槽位缓存是否失效。
    simulation.command("group family clear")
    simulation.command("group family id 1")
    simulation.command("run 0")
    saved_single=args.single;args.single=True
    refreshed=state()
    args.single=saved_single
    if not np.isfinite(np.array(after["rows"])).all(): raise AssertionError("nonfinite trajectory")
    result=dict(passed=True,ranks=comm.size,lammps_args=args.lammps_args,single=args.single,
                source_restart=str(restart) if restart else None,steps=args.steps,
                table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
                force_peak=float(np.max(np.abs(bias_forces))),
                net_bias_force=bias_forces.sum(axis=0).tolist(),finite_difference=details,
                baseline=baseline,frames=frames,refreshed=refreshed)
    if comm.rank==0:
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        # 固定表逐帧偏势可直接用于静态重加权；时间间隔见 steps/脚本。
        (output/"frames.csv").write_text("step,mean_descriptor,bias_energy_eV\n"+
            "".join(f'{frame["step"]},{frame["mean"]:.17g},{frame["energy"]:.17g}\n' for frame in frames))
        print(f"PASS Al3Sc shared frozen: energy reference, FD step sizes={len(details)}, MEAM steps={args.steps}, selection refresh",flush=True)
finally:
    simulation.close()
    simulation.lib.lammps_kokkos_finalize()
