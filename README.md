# ROCm & ComfyUI for 7900xtx
ComfyUI usage and debugging experience on Win or WSL

# News
1. --use-ck-attention is supported, as fast as flash-attn(RDNA4 only)  
2. --enable-dynamic-vram is supported, better vram management( the latest ComfyUI enabled by default, --disable-dynamic-vram needed)  
3. According to forum feedback, rocm10.* has improved by ~40%
4. sageattention-2.2.0 is supported

# Hardware
CPU: 7950X  
GPU: 7900XTX  
RAM: 32GB  
OS: Win11 / WSL  
Python: 3.12  

# Updata ROCm & PyTorch (nightly)
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx1100]==2.11.0+rocm10.1.0a20260815" "torchvision[device-gfx1100]==0.26.0+rocm10.1.0a20260815" "torchaudio==2.11.0+rocm10.1.0a20260815"  
Note: find the latest compiled software that has been approved on https://therock-hud.amd.com/#multi-arch-release  

# Model weight (Recommended)
INT8 ConvRot (best quality-speed balance) > w4a8 (more extreme speed & VRAM savings) > Q4_K_M (fallback compromise) > Others  
Note: w4a8 is my favorite

# Video resolution
480P is my favor. (832x480, 864x480)  
Note: 480x320 for test

# Issues
torch::nms not found - fixed by skipping torch::nms register (modify code according to AI suggestions)  
.onnx can't be detected - fixed by modifying folder_path.py  
INT8-Fast-ROCM crash - fixed by updating the latest packages  

# Reference
https://github.com/patientx-cfz/comfyui-rocm  
https://www.reddit.com/r/ROCm/  
https://github.com/ROCm/TheRock  
https://rocm.docs.amd.com/en/latest/
