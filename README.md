# kalmx — Kalman Filter & RTS Smoother (C++17, Eigen)

A minimal library for linear Gaussian state-space models:
- Kalman Filter (predict/update, batch filtering)
- Rauch–Tung–Striebel (RTS) smoother
- Work in progress: Python bindings via pybind11

## Environment

This project ships with a `spack.yaml` that defines all dependencies (CMake, Eigen, clang-tidy, clang-format, etc.).  
If you don't have Spack installed yet:

```bash
git clone https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
```

Create and activate the environment from `spack.yaml`:
```bash
spack env activate kalmx spack.yaml
```

Install dependencies:
```bash
spack concretize
spack install
```

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If you have `JustFile` installed, you can also simply run
```bash
just configure
just build
```

Additional commands are available through `just`, such as linting & formatting:
```bash
just lint
just fmt
```

## Run demo
```bash
./build/kf_demo
```

<!-- ## Python (optional)
Install `pybind11` (e.g. `pip install pybind11`) then rebuild. A `kalmx_py` module will be produced in `build/`. -->
