#!/usr/bin/env python3
"""Initialize an explicit CUDA device before a Python MPI test imports MPI.

Diagnostic compatibility launcher for OpenMPI 5.0.6's lazy CUDA registration
and finalize path. Does not disable CUDA-aware MPI, suppress sanitizer errors,
or reset/finalize a context owned by the MD engine. Each caller selects the
device within its CUDA_VISIBLE_DEVICES namespace explicitly.
"""
import argparse
import ctypes
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--cudart', default='libcudart.so.12')
    parser.add_argument('script', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.device < 0:
        parser.error('--device must be nonnegative')
    cuda = ctypes.CDLL(args.cudart)
    cuda.cudaSetDevice.argtypes = [ctypes.c_int]
    cuda.cudaSetDevice.restype = ctypes.c_int
    cuda.cudaFree.argtypes = [ctypes.c_void_p]
    cuda.cudaFree.restype = ctypes.c_int
    cuda.cudaGetErrorString.argtypes = [ctypes.c_int]
    cuda.cudaGetErrorString.restype = ctypes.c_char_p
    for operation in (lambda: cuda.cudaSetDevice(args.device), lambda: cuda.cudaFree(None)):
        status = operation()
        if status:
            raise RuntimeError(cuda.cudaGetErrorString(status).decode())
    script = args.script.resolve()
    sys.argv = [str(script), *args.arguments]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
