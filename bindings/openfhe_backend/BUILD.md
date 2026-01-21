# Building the CKKS OpenFHE Backend

This document provides comprehensive instructions for building the `ckks_openfhe_backend` Python extension, which provides pybind11 bindings to OpenFHE's GPU-accelerated CKKS implementation.

## Prerequisites

### Required Software

| Component | Minimum Version | Notes |
|-----------|-----------------|-------|
| **CUDA Toolkit** | 11.0+ | Including nvcc, cublas, curand |
| **Python** | 3.10+ | With development headers |
| **CMake** | 3.18+ | Build system generator |
| **GCC/Clang** | 9.0+/11.0+ | C++17 support required |
| **OpenFHE GPU** | Latest | Must be built first |

### Python Dependencies

```bash
pip install pybind11 scikit-build-core cmake ninja numpy torch
```

## Quick Start

### 1. Build OpenFHE GPU First

The backend requires a pre-built OpenFHE GPU library:

```bash
# From repository root
cd openfhe-gpu-public
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..
```

### 2. Set Environment Variables

```bash
# Required: Tell the linker where to find OpenFHE libraries
export OPENFHE_ROOT="$PWD/openfhe-gpu-public"
export OPENFHE_BUILD_DIR="$OPENFHE_ROOT/build"
export LD_LIBRARY_PATH="$OPENFHE_BUILD_DIR/lib:$OPENFHE_BUILD_DIR/_deps/rmm-build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### 3. Install the Backend

```bash
# Development install (recommended for testing)
pip install -e bindings/openfhe_backend

# OR: Standard install
pip install bindings/openfhe_backend

# OR: Build wheel for distribution
pip wheel bindings/openfhe_backend -w dist/
```

## Build Options

### Custom OpenFHE Location

If OpenFHE is installed in a non-standard location:

```bash
pip install bindings/openfhe_backend \
    --config-settings=cmake.define.OPENFHE_ROOT=/path/to/openfhe-gpu-public \
    --config-settings=cmake.define.OPENFHE_BUILD_DIR=/path/to/openfhe-gpu-public/build
```

### Debug Build

```bash
pip install bindings/openfhe_backend \
    --config-settings=cmake.build-type=Debug
```

### Verbose Build Output

```bash
pip install bindings/openfhe_backend -v
```

## Manual CMake Build

For more control over the build process:

```bash
cd bindings/openfhe_backend

# Configure
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENFHE_ROOT="$PWD/../../openfhe-gpu-public" \
    -DOPENFHE_BUILD_DIR="$PWD/../../openfhe-gpu-public/build"

# Build
cmake --build build -j$(nproc)

# The extension will be in build/ckks_openfhe_backend.cpython-*.so
```

## Verification

### Check Import

```python
python -c "import ckks_openfhe_backend; print('Backend loaded successfully')"
```

### Run Basic Test

```python
python -c "
from ckks import CKKSContext, CKKSConfig
config = CKKSConfig(
    poly_mod_degree=16384,
    coeff_mod_bits=(60, 40, 40, 60),
    scale_bits=40,
    security_level='128_classic',
)
ctx = CKKSContext(config)
print(f'CKKS context created with ring dimension {ctx.ring_dim}')
"
```

### Run Examples

```bash
# GPU benchmark
python bindings/openfhe_backend/examples/bench_gpu.py \
    --slots 1024 --depth 3 --security notset --device cuda

# Encrypted MLP inference
python bindings/openfhe_backend/examples/encrypted_mlp.py --dim 4
```

## Troubleshooting

### Error: `openfhe.h` not found

**Cause**: OpenFHE include directories not found.

**Solution**: Ensure `OPENFHE_ROOT` points to the correct location:

```bash
# Check the path exists
ls $OPENFHE_ROOT/src/pke/include/openfhe.h

# If not found, update OPENFHE_ROOT
export OPENFHE_ROOT=/correct/path/to/openfhe-gpu-public
```

### Error: Cannot find `-lOPENFHEcore`

**Cause**: OpenFHE libraries not built or not in library path.

**Solution**:

```bash
# Verify libraries exist
ls $OPENFHE_BUILD_DIR/lib/libOPENFHE*.so

# If not found, rebuild OpenFHE
cd $OPENFHE_ROOT/build && make -j$(nproc)

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$OPENFHE_BUILD_DIR/lib:$LD_LIBRARY_PATH"
```

### Error: CUDA not found

**Cause**: CUDA Toolkit not installed or not in PATH.

**Solution**:

```bash
# Check CUDA installation
nvcc --version

# If not found, add to PATH (adjust for your CUDA version)
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

### Error: `pybind11` not found

**Solution**:

```bash
pip install pybind11
```

### Runtime Error: Library not found

**Cause**: RPATH not set correctly or libraries moved.

**Solution**: Set `LD_LIBRARY_PATH` before running:

```bash
export LD_LIBRARY_PATH="$OPENFHE_BUILD_DIR/lib:$OPENFHE_BUILD_DIR/_deps/rmm-build:$LD_LIBRARY_PATH"
python your_script.py
```

## Directory Structure

```
bindings/openfhe_backend/
├── BUILD.md                    # This file
├── README.md                   # Usage documentation
├── CMakeLists.txt              # CMake build configuration
├── pyproject.toml              # Python package configuration
├── src/
│   ├── ckks_openfhe_backend.cpp  # C++ pybind11 bindings
│   └── ckks/
│       ├── __init__.py         # Python package init
│       ├── torch_api.py        # High-level PyTorch API
│       └── backends/
│           └── __init__.py
└── examples/
    ├── bench_gpu.py            # GPU performance benchmark
    ├── encrypted_mlp.py        # Basic encrypted inference
    └── encrypted_mlp_trained.py # Train-then-encrypt demo
```

## Performance Tips

1. **Use Release builds** for production (default with pip install)
2. **Set appropriate ring dimension** - larger = more secure but slower
3. **Pre-allocate rotation keys** only for rotations you'll use
4. **Batch operations** when possible to reduce kernel launch overhead

## CI/CD Integration

### GitHub Actions Example

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:11.8.0-devel-ubuntu22.04
    steps:
      - uses: actions/checkout@v4
      
      - name: Install dependencies
        run: |
          apt-get update && apt-get install -y python3-dev python3-pip cmake
          pip install pybind11 scikit-build-core cmake ninja
      
      - name: Build OpenFHE
        run: |
          cd openfhe-gpu-public
          mkdir build && cd build
          cmake .. -DCMAKE_BUILD_TYPE=Release
          make -j$(nproc)
      
      - name: Build backend
        run: |
          export LD_LIBRARY_PATH="$PWD/openfhe-gpu-public/build/lib:$LD_LIBRARY_PATH"
          pip install bindings/openfhe_backend
```

## License

Apache License 2.0
