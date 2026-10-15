@echo off
rem Aura White - one-click Windows installer. Creates a private .venv so nothing else on the PC is touched.
rem   install.bat            -> CPU (or whatever PyTorch pip picks by default)
rem   install.bat cu126      -> NVIDIA GPU build of PyTorch (also: cu124, cu128 ...)
rem   add the word  3d  for the TRUE-3D engine (text / pictures -> real closed asset, runs in 8 GB or less): install.bat cu126 3d
rem   add the word  studio  as the last argument to also install the texturing extras, e.g.  install.bat cu126 studio
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -m venv .venv || (echo Could not create a virtual environment. Install Python 3.9+ from python.org and retry. & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
set STUDIO=0
set GEN3D=0
if /i "%~1"=="3d" set GEN3D=1
if /i "%~2"=="3d" set GEN3D=1
if /i "%~3"=="3d" set GEN3D=1
if /i "%~1"=="studio" set STUDIO=1
if /i "%~2"=="studio" set STUDIO=1
if not "%~1"=="" if /i not "%~1"=="studio" if /i not "%~1"=="3d" (
  echo Installing PyTorch for %~1 ...
  pip install torch --index-url https://download.pytorch.org/whl/%~1 || exit /b 1
)
pip install -e ".[recommended]" || pip install -e .
if "%STUDIO%"=="1" (
  echo Installing the texturing extras ...
  pip install -r requirements-studio.txt || echo Some optional extras failed - the studio uses built-in fallbacks for those.
)
if "%GEN3D%"=="1" (
  echo Installing the true-3D engine ...
  pip install -r requirements-3d.txt || echo Some 3d extras failed - run  run.bat doctor3d  to see what is missing.
  python -m aura_white setup --hunyuan || echo Could not fetch Hunyuan3D-2 - unzip Hunyuan3D-2-main into third_party\ by hand.
  python -m aura_white doctor3d
)
echo.
echo ---- checking the install ----
python -m aura_white doctor
echo.
echo Next:  run.bat make picture.png      (real 3D with back and sides)
echo        run.bat serve        (web app with 3D viewer)
echo        run.bat studio art.png   (sharp textured asset)
echo        run.bat photo.png    (command line)
endlocal
