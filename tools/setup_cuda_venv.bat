@echo off
setlocal
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install --no-cache-dir gsplat
echo.
echo Done.
endlocal
