# cnine

[![Conda CPU build](https://github.com/risi-kondor/cnine/actions/workflows/pytest-cpu.yml/badge.svg)](https://github.com/risi-kondor/cnine/actions/workflows/pytest-cpu.yml)
[![Conda GPU/CUDA build](https://github.com/risi-kondor/cnine/actions/workflows/pytest-gpu.yml/badge.svg)](https://github.com/risi-kondor/cnine/actions/workflows/pytest-gpu.yml)

Lightweight C++ tensor library

Cnine is a simple C++/CUDA tensor library developed by Risi Kondor's group at the University of Chicago.
Cnine is designed to make some of the power of modern GPU architectures accessible directly from C++ code, without relying on complex proprietary libraries.

Documentation for the Python/PyTorch API is at https://risi-kondor.github.io/cnine/.

Cnine is released under the custom noncommercial license included in the file LICENSE.TXT

## Installation

## GPU / CUDA installation

Please install the CUDA toolkit first!

Install with pip (includes PyTorch 2.0+):
```bash
pip install .
```

## CPU installation

For custom PyTorch versions:
1. Install desired PyTorch CPU first
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
2. Then install the build dependencies of cnine manually. (Listed in `pyproject.toml`/`build-system`/`requires`.) Except for `torch` since we manually installed that first.
   ```bash
   pip install scikit-build-core pybind11
   ```
2. Then install cnine, we disable the isolation build, so that module we just manually installed are used as dependencies during the build stage
   ```bash
   pip install --no-build-isolation .
   ```

#### Manual CMake Build

Cnine uses scikit-build-core. To build manually:
```bash
git clone https://github.com/risi-kondor/cnine.git
cd cnine
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/python/installation"
make -j4
```
