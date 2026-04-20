@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_DIR=%CD%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_CONFIG=%VENV_DIR%\pyvenv.cfg"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

if not exist "%VENV_PYTHON%" goto bootstrap
if not exist "%VENV_CONFIG%" goto bootstrap

call "%VENV_PYTHON%" -m quant_pd_framework.gui_launcher
if errorlevel 1 (
  set "EXIT_CODE=!errorlevel!"
  echo.
  echo Failed to launch the Quant Studio GUI from the local virtual environment.
  echo If dependencies changed, run setup_gui.bat to rebuild .venv.
  pause
  endlocal
  exit /b !EXIT_CODE!
)

endlocal
exit /b 0

:bootstrap
echo.
echo Local GUI environment is missing or invalid. Bootstrapping .venv...
call "%~dp0setup_gui.bat"
if errorlevel 1 (
  set "EXIT_CODE=!errorlevel!"
  echo.
  echo GUI setup failed, so the launcher cannot continue.
  pause
  endlocal
  exit /b !EXIT_CODE!
)

if not exist "%VENV_PYTHON%" (
  echo.
  echo Setup completed but the virtual environment Python was not found.
  pause
  endlocal
  exit /b 1
)

echo.
echo Launching Quant Studio from .venv...
call "%VENV_PYTHON%" -m quant_pd_framework.gui_launcher
if errorlevel 1 (
  set "EXIT_CODE=!errorlevel!"
  echo.
  echo Failed to launch the Quant Studio GUI from the bootstrapped environment.
  pause
  endlocal
  exit /b !EXIT_CODE!
)

endlocal
exit /b 0
