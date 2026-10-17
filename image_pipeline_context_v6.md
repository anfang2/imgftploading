# [Context v6] Off-line Vision Inspection & Multi-Model Deblurring System Blueprint

## 1. System Overview & Constraints
- **Target Platform**: Windows 64-bit (`win_amd64`)
- **Python Version**: 3.12 (Strictly enforced)
- **Deployment Environment**: Off-line Closed Corporate Network (폐쇄망 환경)
- **Primary Goal**: Detect blurred BMP images from vision cameras via Folder Monitoring (Watchdog) within milliseconds, and route bad samples to an offline-capable Multi-Model Deep Learning inference processor to deblur images within 4-5 seconds.

## 2. Verified Local File Assets (In-Stock)
The following verified assets are downloaded on the local disk (`C:\Users\A\Downloads\업데이용` or root project) and available for the pipeline:

### 📦 Python Wheel Packages (Dependency Packages)
- `numpy-2.5.3-cp312-cp312-win_amd64.whl`
- `opencv_python-5.0.0.93-cp37-abi3-win_amd64.whl`
- `watchdog-6.0.0-py3-none-win_amd64.whl`
- *(Pre-installed: `torch`, `torchvision` in Python 3.12 environment)*

### 📁 AI Backbone Model Frameworks (Source ZIP)
- `NAFNet-main.zip` (Architecture framework for NAFNet)
- `Restormer-main.zip` (Architecture framework for Restormer)
- `ADNet-master.zip` (Architecture framework for EG-Restormer Edge-guided restoration)

### 🎯 Pre-trained Neural Weights (.pth)
- `NAFNet-GoPro-width32.pth` (67,062 KB - Light-weight Gated-CNN backbone)
- `motion_deblurring.pth` (102,247 KB - Standard Restormer Transformer backbone)
- `egrestormer_gopro.pth` (102,307 KB - Edge-guided structural restoration backbone)

---

## 3. Deployment Directory Structure Strategy
AI Agent must generate paths relative to the project root specified below to prevent environment path collisions:

```text
📁 Project_Root/
│
├── 📜 main.py                         # Unified execution file (Watchdog + Inference Server)
│
├── 📁 offline_wheels/                 # Local directory holding wheel files
│   ├── numpy-2.5.3-cp312-cp312-win_amd64.whl
│   ├── opencv_python-5.0.0.93-cp37-abi3-win_amd64.whl
│   └── watchdog-6.0.0-py3-none-win_amd64.whl
│
├── 📁 models/                         # Extracted framework repositories
│   ├── NAFNet/                        # From NAFNet-main.zip
│   ├── Restormer/                     # From Restormer-main.zip
│   └── EG_Restormer/                  # From ADNet-master.zip
│
└── 📁 weights/                        # Neural checkpoint target destination
    ├── NAFNet-GoPro-width32.pth
    ├── motion_deblurring.pth
    └── egrestormer_gopro.pth
```

---

## 4. Off-line Package Deployment Instructions
To install local dependencies safely within the internal closed grid without downloading from external PyPI indices, use the following PowerShell script command:

```powershell
pip install --no-index --find-links="./offline_wheels" --no-warn-script-location watchdog opencv-python
```

---

## 5. Architectural Pipeline Specifications & Features

### Feature A: Statistical Auto-Calibration Engine
Instead of utilizing static arbitrary threshold metrics, the system runs an automatic statistical baseline profile logic by reading a directory of clear vision targets. It calculates the optimal threshold through standard deviation rules (k-factor constraint):

\[\text{Threshold} = \mu_{\text{Variance}} - (k \times \sigma_{\text{Variance}})\]

- **Strict Mode**: k = 1.0 (High threshold sensitivity, captures minor blurs)
- **Normal Mode**: k = 2.0 (Recommended field production standard)
- **Lenient Mode**: k = 3.0 (Filters extreme structural defects only)

### Feature B: Robust I/O Lock via Decorator Patterns
When the vision camera sensor drops images into the polling grid directory, a file integrity wrapper blocks execution until file sizing stabilizes (preventing partial read crashes due to incomplete storage writing operations).

---

## 6. Base Multi-Model Inference Integration Template (PyTorch 2.6+)
When loading legacy weights or custom structured modules under modern PyTorch iterations on native CPU cores, utilize explicit mapping parameters to bypass pickling environment security exceptions:

```python
import os
import sys
import torch
import numpy as np
import cv2

# Safe loading wrapper supporting backward compatible model structures 
def safe_load_checkpoint(weight_path):
    try:
        # Resolves legacy .tar format loading blocks under PyTorch 2.6+
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        
        if isinstance(checkpoint, dict):
            return checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        return checkpoint
    except Exception as e:
        print(f"[-] Execution Error during safe checkpoint initialization: {str(e)}")
        return None

print("[+] Context Configuration Version 6 Loaded successfully for AI Agent routing.")
```
