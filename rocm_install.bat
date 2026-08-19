@echo off
setlocal enabledelayedexpansion

:: This script must be placed in the root directory of the ComfyUI.
:: ComfyUI-INT8-Fast-ROCM: quantized and load and use the int4(krea2 models is ok and partial Lora support. ltxvideo error)/int8 models
:: CFZ-Caching: save/load conditioning -> Cache your CLIP text encoder output to disk and reload it in future runs - skipping re-encoding entirely.
::: Note: Does not work with models that require VAE input in CLIP text (e.g. qwen-image-edit).
::: nodes: CFZ Save Conditioning, Connect to a CLIP Text Encode node to save conditioning to disk. CFZ Load Conditioning, Load previously saved conditioning by name
::: CUDNN, Simple node to enable or disable MIOpen and its benchmark mode.
::: CUDNN Advanced and MIOpen Nodes
:: ComfyUI-HFRemoteVae: This node allows using Hugginface remote server for latent decoding.
:: CFZ-SwitchMenu: switch menu types between old and new menus
:: https://github.com/Starnodes2024/comfyui-starnodes-modelconverter , Ultimate Model Converter for ComfyUI using comfyui-kitchen - Convert between Transformers, FP32, FP16, FP8. INT8, NVFP4, INT8 Comvrot
:: sage-attention is faster in image/video but some workflows may be black. fast-attention is better at llvm. SpargeAttn?

:: Check python env
python --version
if errorlevel 1 echo Warning: python is not installed, 3.12 is required.

:: 7900xtx belongs to gfx1100
arch=gfx1100
echo GPU Architecture: %arch%

::
:: Installation of the software and configuration of the environment
::
:scratch_installation
:: Install PyTorch
python -m pip install "torch[device-!arch!]" "torchvision[device-!arch!]" torchaudio rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ --no-warn-script-location
if errorlevel 1 goto :install_failed

:: Install popular 3rd-extensions
cd custom_nodes
if not exist comfyui-manager git clone https://github.com/Comfy-Org/ComfyUI-Manager
if not exist CFZ-SwitchMenu git clone https://github.com/patientx/CFZ-SwitchMenu.git
if not exist CFZ-Caching git clone https://github.com/patientx/CFZ-Caching
if not exist ComfyUI-HFRemoteVae git clone https://github.com/kijai/ComfyUI-HFRemoteVae
if not exist ComfyUI-INT8-Fast-ROCM git clone https://github.com/patientx/ComfyUI-INT8-Fast-ROCM
cd ..

:: Install diffusers
python -m pip install diffusers

:: Install triton & sageattention
python -m pip install triton-windows==3.7.0.post26
if errorlevel 1 goto :install_failed

python -m pip install sageattention==1.0.6
if errorlevel 1 goto :install_failed

:: Patching sage-attention
del python_env\Lib\site-packages\sageattention\attn_qk_int8_per_block.py
curl -sL -o python_env\Lib\site-packages\sageattention\attn_qk_int8_per_block.py https://raw.githubusercontent.com/patientx/ComfyUI-Zluda/refs/heads/master/comfy/customzluda/sa/attn_qk_int8_per_block.py
del python_env\Lib\site-packages\sageattention\attn_qk_int8_per_block_causal.py
curl -sL -o python_env\Lib\site-packages\sageattention\attn_qk_int8_per_block_causal.py https://raw.githubusercontent.com/patientx/ComfyUI-Zluda/refs/heads/master/comfy/customzluda/sa/attn_qk_int8_per_block_causal.py
del python_env\Lib\site-packages\sageattention\quant_per_block.py
curl -sL -o python_env\Lib\site-packages\sageattention\quant_per_block.py https://raw.githubusercontent.com/patientx/ComfyUI-Zluda/refs/heads/master/comfy/customzluda/sa/quant_per_block.py

:: Install SpargeAttn
::: https://github.com/jammm/SpargeAttn

:: Install bitsandbytes
python -m pip install https://github.com/0xDELUXA/bitsandbytes_win_rocm/releases/download/0.50.0.dev0-py3.12-rocm7.15-win_amd64_all/bitsandbytes-0.50.0.dev0-cp312-cp312-win_amd64.whl
if errorlevel 1 goto :install_failed

:: Install flashattention (aiter triton backend)
python -m pip install https://github.com/0xDELUXA/flash-attention/releases/download/v2.8.4_win-rocm/flash_attn-2.8.4-py3-none-win_amd64.whl
if errorlevel 1 (
    echo Warning: flash-attention install failed, skipping...
    goto :verify_env
)
python -m pip install https://github.com/0xDELUXA/flash-attention/releases/download/v2.8.4_win-rocm/amd_aiter-0.0.0-py3-none-win_amd64.whl
if errorlevel 1 echo Warning: aiter install failed, flash-attention will not work...

goto :verify_env

::
:: The necessary configurations required for running this ComfyUI locally
::
:run_comfyui_local
.\python_env\scripts\rocm-sdk init >nul 2>&1
if errorlevel 1 (
    echo rocm-sdk init failed. ROCm may not be set up correctly.
)
for /f "delims=" %%i in ('rocm-sdk path --root') do set "HIP_PATH=%%i"
set "ROCM_PATH=%HIP_PATH%"

set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

:: ------------------------- cache and database paths (relative) ------------------------::
set "PYTORCH_TUNABLEOP_CACHE_DIR=%~dp0tunableop-cache"
set "TRITON_CACHE_DIR=%~dp0triton-cache"
:: if you already have a previous triton cache you can define it here so you won't have to rebuild it.

if not exist "%TRITON_CACHE_DIR%" (
    mkdir "%TRITON_CACHE_DIR%"
)

if not exist "%PYTORCH_TUNABLEOP_CACHE_DIR%" (
    mkdir "%PYTORCH_TUNABLEOP_CACHE_DIR%"
)

:: ------------------- CHANGE THESE IF YOU KNOW WHAT YOU ARE DOING --------------------- ::
:: ---------------------- advanced settings (miopen , triton etc.) --------------------- ::
set COMFYUI_ENABLE_MIOPEN=0
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set MIOPEN_FIND_ENFORCE=1
set MIOPEN_FIND_MODE=2
set MIOPEN_DEBUG_DISABLE_FIND_DB=0
set MIOPEN_SEARCH_CUTOFF=1
set MIOPEN_ENABLE_LOGGING=0
set MIOPEN_LOG_LEVEL=0
set MIOPEN_ENABLE_LOGGING_CMD=0
set TRITON_PRINT_AUTOTUNING=0
set TRITON_CACHE_AUTOTUNING=0

:: ------------------- CHANGE THESE IF YOU KNOW WHAT YOU ARE DOING --------------------- ::
:: ----------------- comfyui-rocm STARTUP OPTIONS : modify to your needs --------------- ::

:: triton-backend works for rdna3 & rdna4 BUT the included ComfyUI-INT8-Fast-ROCM crashes comfy with it enabled AND that node is faster than native even on rdna4 so it is better to use than native nodes
:: so triton-backend should be disabled for newer gpu's (it already won't activate for older gens) , if you update triton-windows to >= 3.7.1.post27 it activates itself for rdna3 and above. 
:: if you have rdna3 or rdna4 and despite them being slower, you want to continue using native nodes, uninstall triton-windows. OR let it be disabled and use the custom node.
set PARAMS=--disable-api-nodes --cache-none --disable-smart-memory --disable-pinned-memory --enable-dynamic-vram --use-ck-attention --enable-manager --enable-manager-legacy-ui --disable-triton-backend

:: --------------------------- keeping the necessary packages up-to-date --------------- ::
echo Syncing tracked packages from requirements.txt...
for %%P in (comfyui-frontend-package comfyui-workflow-templates comfyui-embedded-docs comfy-kitchen comfy-aimdo) do (
    for /f "tokens=1,2 delims==" %%A in ('findstr /i "^%%P==" "%~dp0requirements.txt"') do (
        python -m pip show %%A 2>nul | findstr /i "^Version:" > "%TEMP%\pkgver.txt"
        set INSTALLED=
        for /f "tokens=2" %%V in (%TEMP%\pkgver.txt) do set INSTALLED=%%V
        if not "!INSTALLED!"=="%%B" (
            python -m pip install "%%A==%%B" --quiet 2>nul
            echo %%A  !INSTALLED! ^> %%B
        )
    )
)

::
:: How to update ROCM & PyTorch
::
:update_packages
:: Update rocm & pytorch packages
set "UNINSTALL_LIST=rocm rocm-sdk-devel rocm-sdk-core rocm-sdk-libraries torch torchaudio torchvision"
for /f "delims=" %%P in ('python -m pip freeze ^| findstr /I /R "^rocm ^amd-torch ^amd-torchvision"') do (
    for /f "tokens=1 delims=<>=~! " %%N in ("%%P") do (
        echo !UNINSTALL_LIST! | findstr /I /C:"%%N" >nul
        if errorlevel 1 set "UNINSTALL_LIST=!UNINSTALL_LIST! %%N"
    )
)

python -m pip uninstall -y !UNINSTALL_LIST!
python -m pip install --no-cache-dir --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-!arch!]" "torchvision[device-!arch!]" torchaudio rocm-sdk-devel
if errorlevel 1 goto :update_failed

goto :verify_env

::
:: Common
::
:install_failed
echo.
echo ====================================================
echo   Installation Failed!
echo   Check the error messages above for details.
echo ====================================================
goto :end

:update_failed
echo.
echo ====================================================
echo   Update failed!
echo   Check the error messages above for details.
echo ====================================================
goto :end

:verify_env
python -c "import torch; print(f'  PyTorch Version: {torch.__version__}'); print(f'  ROCm Available: {torch.cuda.is_available()}'); print(f'  HIP Version: {torch.version.hip if torch.cuda.is_available() else \"N/A\"}'); print(f'  Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(f'  Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No ROCm device detected\"}')"

:end
pause
exit /b
