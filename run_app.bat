@echo off
setlocal

set "CONDA_ENV_PATH=C:\Users\Asus\miniconda3\envs\surfel_splatting"
set "CUDA_HOME=%CONDA_ENV_PATH%"
set "PATH=%CONDA_ENV_PATH%\bin;%CONDA_ENV_PATH%\Scripts;%PATH%"

cd /d "%~dp0"
echo ========================================================
echo  Starting 2D Gaussian Splatting Web Interface...
echo ========================================================
echo.
"%CONDA_ENV_PATH%\python.exe" app.py
pause
