@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
python -B "%~dp0hdr_to_panorama.py"
if errorlevel 1 set /p "hdr_close=Press Enter to close..."
endlocal
