# 2D Gaussian Splatting: Geometrically Accurate Radiance Field Reconstruction

This directory contains the complete 2D Gaussian Splatting codebase and environment configured for Windows.

## Current Environment Status
- **Python**: Python 3.10.21 (`C:\Users\Asus\miniconda3\envs\surfel_splatting\python.exe`)
- **PyTorch**: 2.5.1+cu121 (CUDA 12.1 acceleration enabled)
- **GPU Detected**: NVIDIA GeForce RTX 3050 4GB Laptop GPU
- **CUDA Compiler**: CUDA 12.1 NVCC installed in Conda environment
- **Dependencies**: `open3d`, `trimesh`, `opencv-python`, `mediapy`, `lpips`, `ninja`, `plyfile`, `scikit-image`, `scipy` installed

---

## Final Step: Compile CUDA Submodules

Because 2D Gaussian Splatting requires compiling custom C++/CUDA kernels (`diff-surfel-rasterization` and `simple-knn`), Microsoft C++ Build Tools (`cl.exe`) is required on Windows.

### 1. If you haven't installed Visual Studio C++ Build Tools:
1. Download the installer from: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. During installation, select **"Desktop development with C++"** and finish install.

### 2. Compile the Submodules:
Double-click `build_submodules.bat` or run:
```powershell
.\build_submodules.bat
```
*(Or open "x64 Native Tools Command Prompt for VS" and run `python setup.py install` inside each submodule folder).*

---

## Running Training & Rendering

### 1. Training on a Scene
Pass your COLMAP dataset folder (e.g. MipNeRF 360 Flowers):
```powershell
.\run_training.bat C:\path\to\flowers output\m360\flowers
```

Or manually:
```powershell
& "C:\Users\Asus\miniconda3\envs\surfel_splatting\python.exe" 2d-gaussian-splatting\train.py -s <dataset_path> -m output/flowers
```

### 2. Rendering & Mesh Extraction
```powershell
.\run_rendering.bat C:\path\to\flowers output\m360\flowers
```

Or manually:
```powershell
& "C:\Users\Asus\miniconda3\envs\surfel_splatting\python.exe" 2d-gaussian-splatting\render.py -s <dataset_path> -m output/flowers --skip_train --skip_test --mesh_res 1024
```
