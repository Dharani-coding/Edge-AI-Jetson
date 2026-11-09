@echo off
setlocal

rem Fixed COCO val2017 subset (500 images, seed 0) with matching annotations.
rem Run from anywhere:  scripts\create_dataset_windows.bat

set "NUM_IMAGES=500"
set "SEED=0"

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"
set "COCO_DIR=%PROJECT_ROOT%\data\coco"
set "LEGACY_ANN_ZIP=%PROJECT_ROOT%\windows\data\coco\annotations_trainval2017.zip"
set "SUBSET=coco_val%NUM_IMAGES%"
set "OUT_DIR=%PROJECT_ROOT%\data\%SUBSET%"
set "ANN_ZIP=%COCO_DIR%\annotations_trainval2017.zip"
set "ANN_JSON=%COCO_DIR%\annotations\instances_val2017.json"

if not exist "%COCO_DIR%" mkdir "%COCO_DIR%"

rem 1. COCO 2017 annotations (about 250 MB, downloaded once)
rem    Flat lines on purpose: inside a ( ... ) block cmd expands %VAR% before
rem    running it, so a "set" there would not reach the tar command.
rem    Windows' own curl/tar are used, because Git's GNU tar cannot read .zip.
if exist "%ANN_JSON%" goto :subset
if not exist "%ANN_ZIP%" if exist "%LEGACY_ANN_ZIP%" set "ANN_ZIP=%LEGACY_ANN_ZIP%"
if exist "%ANN_ZIP%" goto :extract
echo Downloading COCO 2017 annotations...
"%SystemRoot%\System32\curl.exe" -L --fail -o "%ANN_ZIP%" http://images.cocodataset.org/annotations/annotations_trainval2017.zip || goto :error

:extract
echo Extracting instances_val2017.json from %ANN_ZIP% ...
"%SystemRoot%\System32\tar.exe" -xf "%ANN_ZIP%" -C "%COCO_DIR%" annotations/instances_val2017.json || goto :error

:subset
rem 2. Subset annotation JSON + download only the selected images
uv run --project "%PROJECT_ROOT%\windows" --with pycocotools python "%SCRIPT_DIR%make_coco_subset.py" --ann "%ANN_JSON%" --out "%OUT_DIR%" --num %NUM_IMAGES% --seed %SEED% || goto :error

echo.
echo Done: %OUT_DIR%
exit /b 0

:error
echo FAILED (exit code %errorlevel%)
exit /b 1
