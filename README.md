# Image Super-Resolution with Real-ESRGAN

A Python command-line tool that upscales an input image using a pretrained Real-ESRGAN model and saves an enhanced output image.

## Features
- Upscale images with pretrained Real-ESRGAN
- Command-line input and output paths
- Optional GPU acceleration when a compatible PyTorch/CUDA environment is installed
- Simple before/after comparison workflow

## Requirements
- Python 3.10 or 3.11 recommended
- Windows, macOS, or Linux
- Internet access for the first model download
- CPU works but can be slow; compatible NVIDIA GPU recommended for faster inference

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Real-ESRGAN's upstream dependencies can vary by platform. If installation fails, follow the official repository instructions: https://github.com/xinntao/Real-ESRGAN

## Run
Place an image at `inputs/photo.jpg`, then run:
```powershell
python app.py --input inputs/photo.jpg --output outputs/photo_4x.png
```

The script uses the official Real-ESRGAN x4plus pretrained weights. On first run, the model weights are downloaded if not already present.

## Screenshots for your portfolio
Capture genuine results after running the project:
1. `screenshots/01_input.png` — original low-resolution image.
2. `screenshots/02_output.png` — enhanced output.
3. `screenshots/03_comparison.png` — original and output shown side by side.
4. `screenshots/04_terminal.png` — successful command and output path.

Use `Win + Shift + S` on Windows. Add your screenshots to the `screenshots/` directory.

## Limitations
Super-resolution estimates plausible details; it cannot reliably recover details that were never present in the original image. Results vary by image and may introduce artifacts. Do not treat generated details as forensic evidence.

## Resume description
**Image Super-Resolution | Python, PyTorch, Real-ESRGAN**
- Implemented an image enhancement pipeline using a pretrained deep-learning super-resolution model.
- Added command-line input/output handling and a reproducible workflow for comparing source and upscaled images.

## References
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- Project license and model terms should be reviewed before commercial use.
