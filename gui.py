import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


class SuperResolutionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Super-Resolution — Real-ESRGAN")
        self.root.geometry("1100x760")
        self.root.minsize(850, 600)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value=str(Path("outputs") / "enhanced_x4.png"))
        self.model_path = tk.StringVar(value=str(Path("weights") / "RealESRGAN_x4plus.pth"))
        self.tile_size = tk.StringVar(value="0")
        self.original_photo = None
        self.result_photo = None

        main = ttk.Frame(root, padding=14)
        main.pack(fill="both", expand=True)
        ttk.Label(main, text="Image Super-Resolution", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(
            main,
            text="Enhance an image with a pretrained Real-ESRGAN x4 model. Select your model weights before processing.",
            wraplength=1000,
        ).pack(anchor="w", pady=(4, 14))

        form = ttk.Frame(main)
        form.pack(fill="x")
        self.add_path_row(form, 0, "Input image", self.input_path, self.browse_input)
        self.add_path_row(form, 1, "Output image", self.output_path, self.browse_output)
        self.add_path_row(form, 2, "Model weights", self.model_path, self.browse_model)
        ttk.Label(form, text="Tile size (0 = no tiling)").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(form, textvariable=self.tile_size, width=16).grid(row=3, column=1, sticky="w", pady=6)

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=12)
        self.run_button = ttk.Button(buttons, text="Enhance image 4×", command=self.run)
        self.run_button.pack(side="left")
        self.status = tk.StringVar(value="Select an input image and model weights to begin.")
        ttk.Label(buttons, textvariable=self.status).pack(side="left", padx=12)

        compare = ttk.Frame(main)
        compare.pack(fill="both", expand=True, pady=(6, 0))
        original_panel = ttk.LabelFrame(compare, text="Original image", padding=8)
        original_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))
        result_panel = ttk.LabelFrame(compare, text="Enhanced image", padding=8)
        result_panel.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.original_view = ttk.Label(original_panel, text="Choose an image to preview", anchor="center")
        self.original_view.pack(fill="both", expand=True)
        self.result_view = ttk.Label(result_panel, text="Enhanced result will appear here", anchor="center")
        self.result_view.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="Note: AI upscaling may invent or alter fine details; inspect results before relying on them.",
            foreground="#555",
        ).pack(anchor="w", pady=(10, 0))

    def add_path_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(parent, text="Browse…", command=command).grid(row=row, column=2, pady=5)
        parent.columnconfigure(1, weight=1)

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.input_path.set(path)
            try:
                self.original_photo = self.make_preview(Image.open(path).convert("RGB"))
                self.original_view.configure(image=self.original_photo, text="")
                self.result_view.configure(image="", text="Enhanced result will appear here")
                self.result_photo = None
            except Exception as exc:
                messagebox.showerror("Image error", str(exc))

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save enhanced image",
            defaultextension=".png",
            initialfile="enhanced_x4.png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")],
        )
        if path:
            self.output_path.set(path)

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Real-ESRGAN model weights",
            filetypes=[("PyTorch weights", "*.pth"), ("All files", "*.*")],
        )
        if path:
            self.model_path.set(path)

    @staticmethod
    def make_preview(image):
        image.thumbnail((500, 440))
        return ImageTk.PhotoImage(image.copy())

    def run(self):
        input_path = Path(self.input_path.get().strip())
        output_path = Path(self.output_path.get().strip())
        model_path = Path(self.model_path.get().strip())
        try:
            tile = int(self.tile_size.get())
            if tile < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid tile size", "Tile size must be an integer greater than or equal to 0.")
            return
        if not input_path.is_file():
            messagebox.showerror("Missing input", "Choose a valid input image.")
            return
        if not model_path.is_file():
            messagebox.showerror(
                "Missing model weights",
                "Choose RealESRGAN_x4plus.pth. Download the official x4plus weights from the Real-ESRGAN project and select the file here.",
            )
            return

        self.run_button.configure(state="disabled")
        self.status.set("Loading model and enhancing image…")
        threading.Thread(
            target=self.enhance_worker,
            args=(input_path, output_path, model_path, tile),
            daemon=True,
        ).start()

    def enhance_worker(self, input_path, output_path, model_path, tile):
        try:
            import numpy as np
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            image = Image.open(input_path).convert("RGB")
            image_np = np.asarray(image)[:, :, ::-1].copy()
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4
            )
            upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),
            )
            output_bgr, _ = upsampler.enhance(image_np, outscale=4)
            output_rgb = output_bgr[:, :, ::-1]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(output_rgb).save(output_path)
            result_image = Image.fromarray(output_rgb)
            original_size = image.size
            result_size = result_image.size
            self.root.after(0, self.show_result, result_image, original_size, result_size, str(output_path))
        except Exception as exc:
            self.root.after(0, self.show_error, str(exc))

    def show_result(self, image, original_size, result_size, output_path):
        if not self.root.winfo_exists():
            return
        self.result_photo = self.make_preview(image)
        self.result_view.configure(image=self.result_photo, text="")
        self.status.set(
            f"{original_size[0]}×{original_size[1]} → {result_size[0]}×{result_size[1]}. Saved: {output_path}"
        )
        self.run_button.configure(state="normal")
        messagebox.showinfo("Enhancement complete", f"Saved enhanced image to:\n{output_path}")

    def show_error(self, error):
        if not self.root.winfo_exists():
            return
        self.status.set("Enhancement failed.")
        self.run_button.configure(state="normal")
        messagebox.showerror("Enhancement error", error)


if __name__ == "__main__":
    root = tk.Tk()
    app = SuperResolutionGUI(root)
    root.mainloop()
