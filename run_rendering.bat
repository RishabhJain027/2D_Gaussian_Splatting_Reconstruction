@echo off
setlocal

set "CONDA_ENV_PATH=C:\Users\Asus\miniconda3\envs\surfel_splatting"
set "CUDA_HOME=%CONDA_ENV_PATH%"
set "PATH=%CONDA_ENV_PATH%\bin;%CONDA_ENV_PATH%\Scripts;%PATH%"

if "%~1"=="" (
    echo Usage: run_rendering.bat ^<path_to_dataset^> [output_path]
    echo Example: run_rendering.bat C:\datasets\flowers output\m360\flowers
    pause
    exit /b 1
)

set "DATASET=%~1"
set "OUTPUT=%~2"
if "%OUTPUT%"=="" set "OUTPUT=output\flowers"

cd /d "%~dp02d-gaussian-splatting"
echo [*] Starting rendering on dataset: %DATASET%
echo [*] Model checkpoint directory: %OUTPUT%
echo.

"%CONDA_ENV_PATH%\python.exe" render.py -s "%DATASET%" -m "%OUTPUT%" --skip_train --skip_test --mesh_res 1024
pause
