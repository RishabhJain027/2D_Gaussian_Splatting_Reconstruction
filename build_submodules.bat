@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  2D Gaussian Splatting - Submodule Builder (Windows)
echo ========================================================

set "CONDA_ENV_PATH=C:\Users\Asus\miniconda3\envs\surfel_splatting"
set "CUDA_HOME=%CONDA_ENV_PATH%"
set "PATH=%CONDA_ENV_PATH%\bin;%CONDA_ENV_PATH%\Scripts;%PATH%"

where cl.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] cl.exe (MSVC C++ Compiler) was not found in PATH.
    echo.
    echo Please either:
    echo  1. Open the "x64 Native Tools Command Prompt for VS 2022" and run this script, OR
    echo  2. Install Microsoft C++ Build Tools from:
    echo     https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo     (Ensure "Desktop development with C++" workload is selected)
    echo.
    pause
    exit /b 1
)

echo [*] Using Python: %CONDA_ENV_PATH%\python.exe
echo [*] Using CUDA_HOME: %CUDA_HOME%
echo.

cd /d "%~dp02d-gaussian-splatting\submodules\diff-surfel-rasterization"
echo [*] Building diff-surfel-rasterization...
"%CONDA_ENV_PATH%\python.exe" setup.py install
if %errorlevel% neq 0 (
    echo [ERROR] Failed to build diff-surfel-rasterization.
    pause
    exit /b %errorlevel%
)

cd /d "%~dp02d-gaussian-splatting\submodules\simple-knn"
echo.
echo [*] Building simple-knn...
"%CONDA_ENV_PATH%\python.exe" setup.py install
if %errorlevel% neq 0 (
    echo [ERROR] Failed to build simple-knn.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo [SUCCESS] Submodules compiled and installed successfully!
echo ========================================================
pause
