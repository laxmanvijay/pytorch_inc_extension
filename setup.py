import os
import torch
from setuptools import setup
from torch.utils import cpp_extension
import platform

system = platform.system()
architecture = platform.machine()
triplet = None

if system == "Linux":
    triplet = "x64-linux"
elif system == "Darwin":
    triplet = "arm64-osx" if architecture == "arm64" else "x64-osx"
elif system == "Windows":
    triplet = "x64-windows"
else:
    raise ValueError(f"Unsupported platform: {system} on {architecture}")

vcpkg_installed = "vcpkg_installed"

sources = ["src/inc_backend.cpp"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/", f"{os.path.dirname(os.path.abspath(__file__))}/{vcpkg_installed}/{triplet}/include"]

library_dirs = [f"{vcpkg_installed}/{triplet}/lib/"]
libraries = ["fmt"]

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="inc_collectives",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs = library_dirs,
        libraries = libraries,
    )
else:
    module = cpp_extension.CppExtension(
        name="inc_collectives",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs = library_dirs,
        libraries = libraries,
    )

setup(
    name="inc_collectives",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)