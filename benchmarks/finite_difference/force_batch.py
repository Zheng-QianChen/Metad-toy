#!/usr/bin/env python3
"""Verify Graph output batching against the real LAMMPS adapter.

The fixture is an al3sc.py directory (initial.data, shared.table, in.lammps).
Each MPI process checks its own communication counts; forces and virials are
checked against an independent global-ID sum in the C++ probe.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
from mpi4py import MPI
from lammps import lammps

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("plugin","probe","fixture","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fix",default="bias")
    parser.add_argument("--single-center",action="store_true",help="Only atom 1 is a CV center; exercise empty ranks")
    parser.add_argument("--lammps-args",default="")
    parser.add_argument("--expected-submissions",type=int,default=2)
    args=parser.parse_args()
    for name in ("plugin","probe","fixture","output"):
        setattr(args,name,getattr(args,name).resolve())
    comm=MPI.COMM_WORLD
    if comm.rank==0:
        args.output.mkdir(parents=True,exist_ok=False)
        for name in ("initial.data","shared.table"):
            if (args.fixture/name).exists():
                shutil.copy2(args.fixture/name,args.output/name)
    comm.Barrier()
    os.chdir(args.output)
    configuration=(args.fixture/"in.lammps").read_text()
    configuration="\n".join(f"plugin load {args.plugin}" if line.startswith("plugin load ") and Path(shlex.split(line)[2]).name == "fix_crystallize_plugin.so" else line
                            for line in configuration.splitlines())+"\n"
    if args.single_center:
        configuration=configuration.replace("group Sc type 2","group Sc id 1")
        configuration=configuration.replace("group family type 2","group family id 1")
    simulation=lammps(cmdargs=["-screen","none","-log","log.lammps",*shlex.split(args.lammps_args)],comm=comm)
    try:
        simulation.commands_string(configuration)
        probe=ctypes.CDLL(str(args.probe))
        probe.force_batch_error.restype=ctypes.c_char_p
        probe.force_batch_stats.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_uint64)]
        probe.force_batch_stats.restype=ctypes.c_int
        probe.force_batch_check.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int,ctypes.c_int,
                                         ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
        probe.force_batch_check.restype=ctypes.c_int
        def stats():
            output=(ctypes.c_uint64*3)()
            if probe.force_batch_stats(simulation.lmp,args.fix.encode(),output):
                raise RuntimeError(probe.force_batch_error().decode())
            return list(output)
        before=stats()
        simulation.command("run 0")
        after=stats()
        difference=[b-a for a,b in zip(before,after)]
        if difference!=[args.expected_submissions,1,1]:
            raise AssertionError(("Graph submissions/reverse/writeback",difference))
        cases=[]
        for full9 in (0,1):
            for soa in (0,1):
                for cuda_memory in (0,1,2):
                    error=ctypes.c_double()
                    if probe.force_batch_check(simulation.lmp,args.fix.encode(),full9,soa,
                                               cuda_memory,ctypes.byref(error)):
                        raise RuntimeError(probe.force_batch_error().decode())
                    cases.append(dict(components=9 if full9 else 6,layout="SoA" if soa else "AoS",
                                      memory=("Host","Cuda","Mixed")[cuda_memory],max_error=error.value))
        # Direct probe restores MD arrays; a fresh graph evaluation must again form one clean batch.
        before=stats()
        simulation.command("run 0")
        difference_after=[b-a for a,b in zip(before,stats())]
        if difference_after!=[args.expected_submissions,1,1]:
            raise AssertionError(("repeat evaluation",difference_after))
        count=simulation.extract_setting("nlocal")
        center_count=int((simulation.numpy.extract_atom("id")[:count]==1).sum()) if args.single_center else None
        all_results=comm.gather(dict(rank=comm.rank,owned_centers=center_count,graph_counts=difference,
            repeated_counts=difference_after,cases=cases),root=0)
        if comm.rank==0:
            result=dict(passed=True,ranks=comm.size,results=all_results,
                        plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest())
            (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
            print(f"PASS: {comm.size} ranks; Graph {difference}; 12 batch layouts/lifecycles",flush=True)
    finally:
        simulation.close()
        simulation.lib.lammps_kokkos_finalize()
if __name__=="__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
