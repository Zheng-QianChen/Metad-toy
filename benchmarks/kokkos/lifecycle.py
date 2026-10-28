#!/usr/bin/env python3
"""复用有限差分初态，验证 Kokkos 力回写、连续积分、排序、组变化和逐原子输出。

先运行 benchmarks/finite_difference/run.py 生成 nonlinear_graph_np1 夹具。
每种后端在独立目录启动；result.json 中保存完整状态，便于跨后端比较。
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shlex
import numpy as np
from mpi4py import MPI
from lammps import lammps

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--probe", type=Path, required=True)
parser.add_argument("--plugin", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--lammps-args", default="")
parser.add_argument("--newton", choices=["on", "off"], default="on")
args = parser.parse_args()
comm = MPI.COMM_WORLD
output = args.output.resolve()
if comm.rank == 0:
    output.mkdir(parents=True, exist_ok=False)
comm.Barrier()
configuration = (args.fixture.resolve()/"in.lammps").read_text()
# 生产力由 lj/cut(/kk) 提供；CV 半径仍大于 pair 半径，覆盖自定义邻居请求。
configuration = configuration.replace("STRU_FACTOR all ", "STRU_FACTOR tiny ")
configuration = configuration.replace("pair_style zero 2.5", "pair_style lj/cut 3.0")
configuration = configuration.replace("pair_coeff * *", "pair_coeff * * 0.1 2.4")
configuration = configuration.replace("atom_modify map array sort 0 0.0", "atom_modify map array sort 5 0.0")
configuration = configuration.replace("read_data ", "newton "+args.newton+"\nread_data ", 1)
configuration = configuration.replace("neigh_modify every 1 delay 0 check yes",
                                       "neigh_modify every 1 delay 0 check no")
configuration = "\n".join("plugin load "+str(args.plugin.resolve()) if line.startswith("plugin load ")
                           else line for line in configuration.splitlines())
os.chdir(output)
simulation = lammps(cmdargs=["-screen","none","-log","log.lammps",
                            *shlex.split(args.lammps_args)], comm=comm)
try:
    simulation.commands_string(configuration)
    # 独立 LJ 力参考：防止普通与 Kokkos 两端一起遗漏 pair 力而仍通过交叉比较。
    count = simulation.extract_setting("nlocal")
    local = np.column_stack([simulation.numpy.extract_atom("id")[:count],
                             simulation.numpy.extract_atom("x")[:count],
                             simulation.numpy.extract_atom("f")[:count]])
    rows = np.concatenate(comm.allgather(local))
    rows = rows[np.argsort(rows[:,0])]
    reference_force = np.zeros((len(rows),3))
    for first in range(len(rows)):
        for second in range(first+1,len(rows)):
            delta = rows[first,1:4]-rows[second,1:4]
            delta -= np.rint(delta/8.1)*8.1
            distance_sq = delta@delta
            if distance_sq < 3.0**2:
                sixth = (2.4**2/distance_sq)**3
                force = 48*.1*sixth*(sixth-.5)/distance_sq*delta
                reference_force[first] += force
                reference_force[second] -= force
    pair_force_error = float(np.max(np.abs(rows[:,4:7]-reference_force)))
    assert pair_force_error < 1e-11, ("LJ pair force missing or incorrect", pair_force_error)
    probe = ctypes.CDLL(str(args.probe.resolve()))
    probe.fd_add_hill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    probe.fd_add_hill.restype = ctypes.c_int
    probe.kk_position_uploads.argtypes = [ctypes.c_void_p]
    probe.kk_position_uploads.restype = ctypes.c_ulonglong
    group_ids = [1,5,9]
    dimensions = 2
    centers = [simulation.extract_fix("m",0,1,nrow=i)+.06*(-1)**i for i in range(dimensions)]
    assert probe.fd_add_hill(simulation.lmp, (ctypes.c_double*dimensions)(*centers)) == 0
    # 所有阵列输出通过 Host compute 请求，压力检查使用逐原子 virial 的和。
    simulation.commands_string("""
compute cv_multi all metad/atom m l6.stein_q structure.stru_f css_ternary.css
compute cv_group tiny metad/atom m structure.stru_f
compute stress all stress/atom NULL virial
compute stress_sum all reduce sum c_stress[1] c_stress[2] c_stress[3] c_stress[4] c_stress[5] c_stress[6]
thermo_style custom step f_m[1] f_m[2] f_m c_pressure[1] c_pressure[2] c_pressure[3] c_pressure[4] c_pressure[5] c_pressure[6] c_stress_sum[1] c_stress_sum[2] c_stress_sum[3] c_stress_sum[4] c_stress_sum[5] c_stress_sum[6]
thermo 50
timestep 0.0005
velocity all set -3.0 -3.0 -3.0
fix integrate all nve
run 0
""")
    def capture():
        count = simulation.extract_setting("nlocal")
        ids = simulation.numpy.extract_atom("id")[:count].copy()
        values = simulation.numpy.extract_compute("cv_multi",1,2)[:count].copy()
        group = simulation.numpy.extract_compute("cv_group",1,1)[:count].copy()
        # 检查掩码而非仅比较两个后端，避免双方同样漏掉 group 筛选。
        expected = np.where(np.isin(ids,group_ids), values[:,1], 0.0)
        assert np.allclose(group,expected,rtol=0,atol=1e-12)
        rows = np.column_stack([ids,simulation.numpy.extract_atom("x")[:count],
                                 simulation.numpy.extract_atom("f")[:count],values,group,
                                 simulation.numpy.extract_atom("type")[:count]])
        rows = np.concatenate(comm.allgather(rows))
        rows = rows[np.argsort(rows[:,0])]
        pressure = simulation.numpy.extract_compute("pressure",0,1).copy()
        stress = simulation.numpy.extract_compute("stress_sum",0,1).copy()
        assert np.max(np.abs(stress+pressure*simulation.get_thermo("vol"))) < 2e-9
        assert np.isfinite(rows).all()
        return dict(rows=rows.tolist(), values=[simulation.extract_fix("m",0,1,nrow=i) for i in range(dimensions)],
                    energy=simulation.extract_fix("m",0,0), virial=stress.tolist(),
                    owned_counts=comm.allgather(count), owned_ids=comm.allgather(sorted(map(int,ids))))
    states = [capture()]
    # 多次连续 Device 积分；足够位移跨越 MPI 子域边界，且定期排序。
    simulation.command("run 200")
    states.append(capture())
    if comm.size > 1:
        assert states[0]["owned_ids"] != states[1]["owned_ids"], "fixture did not migrate atoms"
    # 同步 Host 修改后的类型和 group，在同一 timestep 重新装配并计算。
    group_ids = [1,5,17]
    simulation.commands_string("set atom 1 type 3\ngroup tiny clear\ngroup tiny id 1 5 17\nrun 0")
    states.append(capture())
    assert states[0]["rows"][0][-1] == 2 and states[-1]["rows"][0][-1] == 3
    before = states[-1]
    simulation.command("run 0")
    after = capture()
    assert np.allclose(before["rows"],after["rows"],rtol=0,atol=1e-10)
    assert np.max(np.abs(np.array(states[1]["rows"])[:,1:4]-np.array(states[0]["rows"])[:,1:4])) > .1
    uploads = probe.kk_position_uploads(simulation.lmp)
    if simulation.extract_setting("kokkos_active"):
        assert uploads == 0, ("Kokkos coordinates used Host upload", uploads)
    else:
        assert uploads > 0, "ordinary LAMMPS coordinate staging was not exercised"
    if comm.rank == 0:
        result = dict(passed=True, ranks=comm.size, args=args.lammps_args, newton=args.newton, position_uploads=int(uploads), pair_force_error=pair_force_error, states=states)
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        print("PASS: trajectory, force, virial, compute output, sorting and refresh",flush=True)
finally:
    simulation.close()
    simulation.lib.lammps_kokkos_finalize()