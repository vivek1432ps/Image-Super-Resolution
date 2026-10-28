# Image Super-Resolution with Real-ESRGAN

A Python image-enhancement project using a pretrained Real-ESRGAN x4plus model. Includes a graphical user interface for selecting images, choosing model weights, processing images, and comparing the original with the enhanced result.

## Features
- **GUI:** input/output/model file pickers and side-by-side image previews
- 4× image upscaling with pretrained Real-ESRGAN x4plus
- Optional tile size to reduce memory use on large images
- Optional GPU acceleration when a compatible PyTorch/CUDA environment is installed
- Existing command-line workflow remains available

## Requirements
- Python 3.10 or 3.11 recommended
- Windows, macOS, or Linux
- CPU works but can be slow; compatible NVIDIA GPU recommended for faster inference

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Real-ESRGAN and BasicSR dependencies can vary by platform. If installation fails, follow the official instructions: https://github.com/xinntao/Real-ESRGAN

## Download the model weights
The app expects the official `RealESRGAN_x4plus.pth` weights. Download them from the official Real-ESRGAN project/releases and save them as `weights/RealESRGAN_x4plus.pth`, or browse to the downloaded `.pth` file inside the GUI. Do not download model weights from untrusted mirrors.

## Launch the graphical interface
```powershell
python gui.py
```
1. Choose an input image.
2. Choose where the enhanced image should be saved.
3. Browse to `RealESRGAN_x4plus.pth`.
4. Keep tile size at `0` initially. If you run out of GPU/CPU memory, try a tile size such as `256`.
5. Click **Enhance image 4×** and compare the preview panels.

## Command-line alternative
```powershell
python app.py --input inputs/photo.jpg --output outputs/photo_4x.png
``
Optional flags: `--model weights/RealESRGAN_x4plus.pth` and `--tile 256`.

## Screenshots for your portfolio
Capture real results after running the project:
1. `screenshots/01_gui_original.png` — GUI with original image loaded.
2. `screenshots/02_gui_result.png` — original and enhanced previews.
3. `screenshots/03_output.png` — saved enhanced image.
4. `screenshots/04_terminal.png` — successful run or completion message.

Use `Win + Shift + S` on Windows. Add your screenshots to the `screenshots/` directory.

## Limitations
Super-resolution estimates plausible details; it cannot reliably recover details that were never present in the original image. Results vary by image and may introduce artifacts. Do not treat generated details as forensic evidence.

## Resume description
**Image Super-Resolution | Python, Tkinter, PyTorch, Real-ESRGAN**
- Built a GUI image enhancement tool with image selection, model configuration, side-by-side preview, and 4× output export.
- Integrated a pretrained deep-learning super-resolution model with optional tiled inference and GPU acceleration.

## References
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- Review upstream license and model terms before commercial use.
