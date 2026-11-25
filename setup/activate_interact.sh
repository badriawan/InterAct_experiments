# Source this file after `conda activate interact`.
if [ "${CONDA_DEFAULT_ENV:-}" != "interact" ]; then
    echo 'Activate the interact Conda environment first.' >&2
    return 1
fi
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PYOPENGL_PLATFORM="egl"
export CMAKE_POLICY_VERSION_MINIMUM="3.5"
