@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_DIR=%CD%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_CONFIG=%VENV_DIR%\pyvenv.cfg"
set "BOOTSTRAP_TEMP=%CD%\.setup_tmp"
set "BOOTSTRAP_PYTHON="
set "BOOTSTRAP_PIP_WHEEL="

if not exist "%BOOTSTRAP_TEMP%" mkdir "%BOOTSTRAP_TEMP%"
if errorlevel 1 (
  set "EXIT_CODE=!errorlevel!"
  echo.
  echo Failed to create the local bootstrap temp directory.
  pause
  endlocal
  exit /b !EXIT_CODE!
)
set "TMP=%BOOTSTRAP_TEMP%"
set "TEMP=%BOOTSTRAP_TEMP%"

where py >nul 2>nul
if %errorlevel%==0 (
  set "BOOTSTRAP_PYTHON=py -3"
  goto found_python
)

where python >nul 2>nul
if %errorlevel%==0 (
  set "BOOTSTRAP_PYTHON=python"
  goto found_python
)

echo.
echo Could not find Python on PATH.
echo Install Python 3.11+ and rerun this script.
pause
endlocal
exit /b 1

:found_python
for /f "usebackq delims=" %%i in (`%BOOTSTRAP_PYTHON% -c "import ensurepip; from pathlib import Path; bundled = Path(ensurepip.__file__).resolve().parent / '_bundled'; print(next(iter(sorted(bundled.glob('pip-*.whl'))), ''))"`) do set "BOOTSTRAP_PIP_WHEEL=%%i"
if not exist "%BOOTSTRAP_PIP_WHEEL%" (
  echo.
  echo Could not locate the bundled pip wheel for the bootstrap Python.
  pause
  endlocal
  exit /b 1
)

if exist "%VENV_DIR%" (
  if not exist "%VENV_PYTHON%" goto rebuild_venv
  if not exist "%VENV_CONFIG%" goto rebuild_venv
)

if not exist "%VENV_PYTHON%" (
  echo.
  echo Creating local virtual environment in .venv...
  call %BOOTSTRAP_PYTHON% -m venv ".venv"
  if errorlevel 1 (
    set "EXIT_CODE=!errorlevel!"
    echo.
    echo Failed to create the local virtual environment.
    pause
    endlocal
    exit /b !EXIT_CODE!
  )
) else (
  echo.
  echo Reusing existing .venv...
)

echo.
echo Ensuring pip is available inside .venv...
call %BOOTSTRAP_PYTHON% -m pip --python "%VENV_PYTHON%" install "%BOOTSTRAP_PIP_WHEEL%"
if errorlevel 1 (
  set "EXIT_CODE=!errorlevel!"
  echo.
  echo Failed while bootstrapping pip inside .venv.
  pause
  endlocal
  exit /b !EXIT_CODE!
)

echo.
echo Installing GUI dependencies into .venv...
call "%VENV_PYTHON%" -m pip install --disable-pip-version-check -e ".[dev,gui]"
if errorlevel 1 (
  echo.
  echo Full dependency installation failed.
  echo Checking whether the bootstrap Python already has the required GUI packages...
  call %BOOTSTRAP_PYTHON% -c "import joblib, openpyxl, pandas, plotly, sklearn, statsmodels, streamlit, xgboost"
  if errorlevel 1 (
    set "EXIT_CODE=!errorlevel!"
    echo.
    echo Failed while installing project dependencies into .venv.
    pause
    endlocal
    exit /b !EXIT_CODE!
  )

  call %BOOTSTRAP_PYTHON% -c "import site; from pathlib import Path; paths = [Path(r'%CD%') / 'src', *[Path(path) for path in site.getsitepackages()], Path(site.getusersitepackages())]; target = Path(r'%VENV_DIR%') / 'Lib' / 'site-packages' / 'quant_pd_launcher_fallback.pth'; target.write_text(''.join(f'{path}\n' for path in paths if path.exists()), encoding='utf-8')"
  if errorlevel 1 (
    set "EXIT_CODE=!errorlevel!"
    echo.
    echo Failed while creating the fallback site-packages bridge for .venv.
    pause
    endlocal
    exit /b !EXIT_CODE!
  )

  echo.
  echo Falling back to packages already available in the bootstrap Python environment.
  echo launch_gui.bat will use the local source tree and bridged site-packages.
)

echo.
echo GUI environment is ready.
echo Launch with launch_gui.bat
if exist "%BOOTSTRAP_TEMP%" rmdir /s /q "%BOOTSTRAP_TEMP%" >nul 2>nul
endlocal
exit /b 0

:rebuild_venv
echo.
echo Existing .venv is incomplete or invalid. Rebuilding it...
rmdir /s /q "%VENV_DIR%"
if exist "%VENV_DIR%" (
  echo.
  echo Failed to remove the broken .venv directory.
  echo Close any terminals using .venv and rerun setup_gui.bat.
  pause
  endlocal
  exit /b 1
)
goto found_python
