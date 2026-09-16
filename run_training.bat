@echo off
setlocal

set "CONDA_ENV_PATH=C:\Users\Asus\miniconda3\envs\surfel_splatting"
set "CUDA_HOME=%CONDA_ENV_PATH%"
set "PATH=%CONDA_ENV_PATH%\bin;%CONDA_ENV_PATH%\Scripts;%PATH%"

if "%~1"=="" (
    echo Usage: run_training.bat ^<path_to_dataset^> [output_path]
    echo Example: run_training.bat C:\datasets\flowers output\m360\flowers
    pause
    exit /b 1
)

set "DATASET=%~1"
set "OUTPUT=%~2"
if "%OUTPUT%"=="" set "OUTPUT=output\flowers"

cd /d "%~dp02d-gaussian-splatting"
echo [*] Starting training on dataset: %DATASET%
echo [*] Output directory: %OUTPUT%
echo.

"%CONDA_ENV_PATH%\python.exe" train.py -s "%DATASET%" -m "%OUTPUT%"
pause
