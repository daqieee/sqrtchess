@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "SCRIPT=gui\sqrtchess_gui.py"
if not exist "%SCRIPT%" (
  echo 找不到 %SCRIPT%，请在项目根目录运行本脚本。
  pause
  exit /b 1
)
where pythonw >nul 2>nul
if %errorlevel%==0 (
  start "" pythonw "%SCRIPT%"
  exit /b 0
)
where py >nul 2>nul
if %errorlevel%==0 (
  start "" py -3w "%SCRIPT%"
  exit /b 0
)
where python >nul 2>nul
if %errorlevel%==0 (
  start "" python "%SCRIPT%"
  exit /b 0
)
echo 没有找到 Python，请先安装 Python 3.8+ 并勾选 "Add Python to PATH"。
pause
