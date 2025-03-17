# cnine

[![Conda CPU build](https://github.com/risi-kondor/cnine/actions/workflows/pytest-cpu.yml/badge.svg)](https://github.com/risi-kondor/cnine/actions/workflows/pytest-cpu.yml)
[![Conda GPU/CUDA build](https://github.com/risi-kondor/cnine/actions/workflows/pytest-gpu.yml/badge.svg)](https://github.com/risi-kondor/cnine/actions/workflows/pytest-gpu.yml)

Lightweight C++ tensor library

Cnine is a simple C++/CUDA tensor library developed by Risi Kondor's group at the University of Chicago.
Cnine is designed to make some of the power of modern GPU architectures accessible directly from C++ code, without relying on complex proprietary libraries. 

Documentation for the Python/PyTorch API is at https://risi-kondor.github.io/cnine/.

Cnine is released under the custom noncommercial license included in the file LICENSE.TXT 

## Installation

### Basic Installation  
Create and activate a virtual environment:  
```bash
python -m venv /path/to/venv
source /path/to/venv/bin/activate
```

Install with pip (includes PyTorch 2.0+):  
```bash
pip install .
```

#### Advanced Installation

For custom PyTorch versions:  
1. Install desired PyTorch first:  
   ```bash
   pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```
2. Then install cnine:  
   ```bash
   pip install .
   ```

#### Manual CMake Build

Cnine uses scikit-build-core. To build manually:  
```bash
git clone https://github.com/yourusername/cnine.git
cd cnine
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/torch"
make -j4
```

