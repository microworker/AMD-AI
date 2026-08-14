# ROCm & ComfyUI for 7900xtx
relevant doc and configuration

# Hardware
CPU: 7950X  
GPU: 7900XTX  
RAM: 32GB  
OS: Win11 / WSL  

# Updata ROCm & PyTorch (nightly)
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "rocm[libraries,device-gfx1100]==10.1.0a20260813"  
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx1100]" "torchvision[device-gfx1100]" torchaudio "rocm-sdk==10.1.0a20260813"  
Note: find the version you want on https://therock-hud.amd.com/#multi-arch-release  
