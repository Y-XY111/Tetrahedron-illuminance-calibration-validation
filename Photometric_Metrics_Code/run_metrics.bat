@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
python -B "%~dp0hdr_metrics_to_excel.py"
if errorlevel 1 set /p "hdr_close=Press Enter to close..."
endlocal
