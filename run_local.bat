@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] 找不到 Python Launcher。請先安裝 Python 3.11 以上版本。
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo 建立虛擬環境...
  py -3 -m venv .venv
)

echo 安裝/更新套件...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo 更新資料與圖表...
.venv\Scripts\python.exe src\update_chart.py

if errorlevel 1 (
  echo.
  echo [ERROR] 執行失敗，請保留上方錯誤訊息。
  pause
  exit /b 1
)

echo.
echo 完成。請開啟 docs\index.html
start "" docs\index.html
endlocal
