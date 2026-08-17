# ROCm & ComfyUI for 7900xtx
relevant doc and configuration

# Hardware
CPU: 7950X  
GPU: 7900XTX  
RAM: 32GB  
OS: Win11 / WSL  
Python: 3.12  

# Updata ROCm & PyTorch (nightly)
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx1100]==2.11.0+rocm10.1.0a20260815" "torchvision[device-gfx1100]==0.26.0+rocm10.1.0a20260815" "torchaudio==2.11.0+rocm10.1.0a20260815"  
Note: find the latest compiled software that has been approved on https://therock-hud.amd.com/#multi-arch-release  
