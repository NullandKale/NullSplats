@echo off
setlocal EnableDelayedExpansion

rem Portable bundle packer. Copies app, venv, tools, and optional CUDA runtime into ./build/NullSplats-portable.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "VENV=%ROOT%\.venv"
set "BUILD_DIR=%ROOT%\build"
set "OUT_DIR=%BUILD_DIR%\NullSplats-portable"
set "COLMAP_SRC=%ROOT%\tools\colmap"
set "GLOMAP_SRC="
set "CUDA_SRC=%CUDA_PATH%"
if "%CUDA_SRC%"=="" set "CUDA_SRC=%CUDA_HOME%"
set "CUDA_SRC_ARG=%1"
set "ZIP_PATH=%BUILD_DIR%\NullSplats-portable.zip"
if "%SKIP_ZIP%"=="" set "SKIP_ZIP=0"
if "%REQUIRE_CUDA%"=="" set "REQUIRE_CUDA=1"

if not "%CUDA_SRC_ARG%"=="" (
    set "CUDA_SRC=%CUDA_SRC_ARG%"
)

if "%SKIP_CLEAN%"=="" set "SKIP_CLEAN=0"

echo.
echo === NullSplats portable bundle ===
echo Repo root:   %ROOT%
echo Venv:        %VENV%
echo Output:      %OUT_DIR%
echo COLMAP src:  %COLMAP_SRC%
echo GLOMAP src:  %GLOMAP_SRC%
echo CUDA src:    %CUDA_SRC%
echo Skip clean:  %SKIP_CLEAN%
echo Skip zip:    %SKIP_ZIP%
echo Require CUDA copy: %REQUIRE_CUDA%
echo.

if not exist "%VENV%" (
    echo Venv not found at %VENV% ^(populate it first^) & exit /b 1
)

if "%SKIP_CLEAN%"=="0" (
    echo Cleaning build folder...
    if exist "%BUILD_DIR%" rd /s /q "%BUILD_DIR%" >nul 2>&1
)
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%" >nul 2>&1
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%" >nul 2>&1
if "%SKIP_CLEAN%"=="0" if exist "%ZIP_PATH%" del /f /q "%ZIP_PATH%" >nul 2>&1

if not exist "%ROOT%\main.py" (
    echo main.py not found at %ROOT%\main.py & exit /b 1
)

echo Copying package...
robocopy "%ROOT%\nullsplats" "%OUT_DIR%\nullsplats" /mir >nul
copy /y "%ROOT%\main.py" "%OUT_DIR%\" >nul

echo Copying shaders...
robocopy "%ROOT%\nullsplats\ui\shaders" "%OUT_DIR%\nullsplats\ui\shaders" /mir >nul

if exist "%COLMAP_SRC%" (
    echo Copying COLMAP...
    robocopy "%COLMAP_SRC%" "%OUT_DIR%\tools\colmap" /mir >nul
) else (
    echo [warn] COLMAP source not found at %COLMAP_SRC%.
)

echo Copying venv...
robocopy "%VENV%" "%OUT_DIR%\venv" /mir >nul

echo Pruning unused Python packages...
for %%D in ("%OUT_DIR%\venv\Lib\site-packages\tqdm" "%OUT_DIR%\venv\Lib\site-packages\tqdm-*.dist-info" "%OUT_DIR%\venv\Lib\site-packages\tyro" "%OUT_DIR%\venv\Lib\site-packages\tyro-*.dist-info" "%OUT_DIR%\venv\Lib\site-packages\cv2" "%OUT_DIR%\venv\Lib\site-packages\opencv_python-*.dist-info" "%OUT_DIR%\venv\Lib\site-packages\yaml" "%OUT_DIR%\venv\Lib\site-packages\PyYAML-*.dist-info") do (
    if exist "%%~D" rd /s /q "%%~D"
)
for /r "%OUT_DIR%\venv" %%F in (*.pdb *.lib *.exp) do del /f /q "%%F" >nul 2>&1

rem ---- CUDA runtime copy ----
if not exist "%CUDA_SRC%" (
    if "%REQUIRE_CUDA%"=="1" (
        echo [error] CUDA source not found at %CUDA_SRC% and REQUIRE_CUDA=1. Set CUDA_PATH/CUDA_HOME or pass a CUDA path argument, or set REQUIRE_CUDA=0 to skip.
        exit /b 1
    ) else (
        echo [warn] CUDA source not found; bundle will not include CUDA DLLs.
    )
) else (
    echo Copying CUDA runtime from %CUDA_SRC% ...
    set "CUDA_DEST_BIN=%OUT_DIR%\cuda\bin"
    mkdir "%OUT_DIR%\cuda" >nul 2>&1
    mkdir "!CUDA_DEST_BIN!" >nul 2>&1

    set "CUDA_PATTERNS=cudart64_*.dll cublas64_*.dll cublasLt64_*.dll cusparse64_*.dll cusolver64_*.dll cufft64_*.dll curand64_*.dll cudnn64_*.dll nvrtc64_*.dll nvrtc-builtins64_*.dll nvJitLink_*.dll"
    set "CUDA_COPIED=0"

    if exist "%CUDA_SRC%\bin" (
        echo ... copying from "%CUDA_SRC%\bin" to "!CUDA_DEST_BIN!"
        echo Source bin candidates:
        dir /b "%CUDA_SRC%\bin\cud*.dll" "%CUDA_SRC%\bin\nvr*.dll" 2>nul | findstr /r "." || echo ^(none^)
        for %%P in (!CUDA_PATTERNS!) do (
            for /f "delims=" %%F in ('dir /b "%CUDA_SRC%\bin\%%P" 2^>nul') do (
                copy /y "%CUDA_SRC%\bin\%%F" "!CUDA_DEST_BIN!\" >nul
                set "CUDA_COPIED=1"
            )
        )
        if "!CUDA_COPIED!"=="0" (
            echo ... fallback copying all cud*/nv* DLLs
            for /f "delims=" %%F in ('dir /b "%CUDA_SRC%\bin\cud*.dll" "%CUDA_SRC%\bin\nv*.dll" 2^>nul') do (
                copy /y "%CUDA_SRC%\bin\%%F" "!CUDA_DEST_BIN!\" >nul
                set "CUDA_COPIED=1"
            )
        )
    ) else (
        echo [warn] CUDA bin not found at "%CUDA_SRC%\bin"
    )

    if "!CUDA_COPIED!"=="0" (
        if "%REQUIRE_CUDA%"=="1" (
            echo [error] CUDA DLL copy produced no files; ensure CUDA_SRC is correct or set REQUIRE_CUDA=0 to skip. & exit /b 1
        ) else (
            echo [warn] No CUDA DLLs were copied.
        )
    ) else (
        echo CUDA DLLs copied to %OUT_DIR%\cuda
        echo CUDA bin contents:
        dir "!CUDA_DEST_BIN!" 2>nul
    )
)

echo Writing run.bat launcher...
(
    echo @echo off
    echo setlocal
    echo set "APP_DIR=%%~dp0"
    echo if "%%APP_DIR:~-1%%"=="\" set "APP_DIR=%%APP_DIR:~0,-1%%"
    echo set "VIRTUAL_ENV=%%APP_DIR%%\venv"
    echo set "PATH=%%VIRTUAL_ENV%%\Scripts;%%PATH%%"
    echo if exist "%%APP_DIR%%\cuda\bin" set "PATH=%%APP_DIR%%\cuda\bin;%%PATH%%"
    echo if exist "%%APP_DIR%%\cuda\lib\x64" set "PATH=%%APP_DIR%%\cuda\lib\x64;%%PATH%%"
    echo set "CUDA_HOME=%%APP_DIR%%\cuda"
    echo set "CUDA_PATH=%%APP_DIR%%\cuda"
    echo set "COLMAP_ROOT=%%APP_DIR%%\tools\colmap"
    echo if exist "%%COLMAP_ROOT%%\bin" set "PATH=%%COLMAP_ROOT%%\bin;%%PATH%%"
    echo if exist "%%COLMAP_ROOT%%\lib" set "PATH=%%COLMAP_ROOT%%\lib;%%PATH%%"
    echo "%%VIRTUAL_ENV%%\Scripts\python.exe" "%%APP_DIR%%\main.py" %%*
) > "%OUT_DIR%\run.bat"
copy /y "%OUT_DIR%\run.bat" "%BUILD_DIR%\run-portable.bat" >nul 2>&1

echo.
if "%SKIP_ZIP%"=="0" (
    echo Creating zip archive...
    where 7z >nul 2>&1
    if %ERRORLEVEL%==0 (
        7z a -tzip -mx=0 "%ZIP_PATH%" "%OUT_DIR%\*" >nul
    ) else (
        powershell -Command "Compress-Archive -Path '%OUT_DIR%\*' -DestinationPath '%ZIP_PATH%' -CompressionLevel Fastest -Force" >nul 2>&1
    )
    if not exist "%ZIP_PATH%" (
        echo [error] Zip creation failed at %ZIP_PATH% & exit /b 1
    )
) else (
    echo Skipping zip creation.
)

echo.
echo [done] Portable bundle ready at %OUT_DIR%
if "%SKIP_ZIP%"=="0" echo [done] Zip archive created at %ZIP_PATH%
echo Use run.bat (in the bundle) or build\run-portable.bat to launch.
echo Build folder contents:
dir "%BUILD_DIR%" /b
exit /b 0
