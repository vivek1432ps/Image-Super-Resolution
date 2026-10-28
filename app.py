import argparse
from pathlib import Path

import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def main():
    parser = argparse.ArgumentParser(
        description="Upscale an image with pretrained Real-ESRGAN x4plus."
    )
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--output", required=True, help="Path for the enhanced image.")
    parser.add_argument(
        "--model",
        default="weights/RealESRGAN_x4plus.pth",
        help="Path to model weights (download instructions are in README).",
    )
    parser.add_argument("--tile", type=int, default=0, help="Tile size; use 0 for no tiling.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    if not input_path.is_file():
        raise SystemExit(f"Input image not found: {input_path}")
    if not model_path.is_file():
        raise SystemExit(
            f"Model weights not found: {model_path}\n"
            "Download RealESRGAN_x4plus.pth from the official Real-ESRGAN releases "
            "or follow the upstream README, then place it at the model path."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(input_path).convert("RGB")
    import numpy as np

    image_np = np.array(image)[:, :, ::-1].copy()  # RGB -> BGR for BasicSR

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=max(0, args.tile),
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )

    output_bgr, _ = upsampler.enhance(image_np, outscale=4)
    output_rgb = output_bgr[:, :, ::-1]
    Image.fromarray(output_rgb).save(output_path)
    print(f"Input size:  {image.width} x {image.height}")
    result = Image.open(output_path)
    print(f"Output size: {result.width} x {result.height}")
    print(f"Saved enhanced image: {output_path.resolve()}")


if __name__ == "__main__":
    main()
