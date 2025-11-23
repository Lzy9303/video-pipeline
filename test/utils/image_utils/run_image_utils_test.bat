@echo off
setlocal
set DIR=%~dp0
pushd "%DIR%"
set ROOT=%DIR%..\..\..\
set VENV_PY=%ROOT%venv\Scripts\python.exe
if exist "%VENV_PY%" (
  set PY=%VENV_PY%
) else (
  set PY=python
)
"%PY%" -c "import cv2" >nul 2>&1
if errorlevel 1 (
  "%PY%" -m pip install -q opencv-python numpy pytest
)
"%PY%" -m pytest -q "image_utils_test.py"
popd
