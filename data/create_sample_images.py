"""Utility to generate realistic botanical synthetic sample leaf images for interactive testing."""

from pathlib import Path
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUTPUT_DIR = Path("data/sample_leaves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_leaf(
    base_color=(50, 140, 45),
    leaf_shape="ovate",
    lesion_type=None,
    size=(300, 300)
) -> Image.Image:
    """Generate a high-contrast leaf image with synthetic biological features."""
    w, h = size
    img = Image.new("RGB", size, (235, 238, 232)) # Pale neutral background
    draw = ImageDraw.Draw(img)

    # Leaf polygon
    cx, cy = w // 2, h // 2
    points = []
    num_pts = 60
    for i in range(num_pts):
        angle = (2 * math.pi * i) / num_pts
        # Base radius
        if leaf_shape == "ovate":
            r_scale = 1.0 - 0.4 * math.sin(angle)
            r = min(w, h) * 0.38 * r_scale * (1.0 + 0.05 * math.sin(angle * 7))
        elif leaf_shape == "compound":
            r = min(w, h) * 0.35 * (0.8 + 0.25 * math.cos(angle * 3))
        elif leaf_shape == "linear": # corn
            r = min(w, h) * (0.45 if abs(math.cos(angle)) > 0.3 else 0.15)
        else: # palmate (grape)
            r = min(w, h) * 0.36 * (0.7 + 0.3 * abs(math.sin(angle * 2.5)))

        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        points.append((px, py))

    # Fill leaf blade
    draw.polygon(points, fill=base_color, outline=(base_color[0]//2, base_color[1]//2, base_color[2]//2))

    # Central vein and secondary veins
    draw.line([(cx, cy - h*0.35), (cx, cy + h*0.38)], fill=(min(255, base_color[0]+30), min(255, base_color[1]+40), base_color[2]), width=3)
    for dy in range(int(-h*0.25), int(h*0.3), 25):
        draw.line([(cx, cy + dy), (cx - w*0.22, cy + dy - 15)], fill=(min(255, base_color[0]+25), min(255, base_color[1]+35), base_color[2]), width=2)
        draw.line([(cx, cy + dy), (cx + w*0.22, cy + dy - 15)], fill=(min(255, base_color[0]+25), min(255, base_color[1]+35), base_color[2]), width=2)

    # Apply specific pathology symptoms
    if lesion_type == "target_rings": # Early blight concentric rings
        for (lx, ly, max_rad) in [(cx - 35, cy - 20, 24), (cx + 40, cy + 30, 28), (cx - 10, cy + 50, 18)]:
            for rad in range(max_rad, 4, -4):
                col = (65 + rad*2, 45 + rad, 20) if rad % 8 == 0 else (120, 85, 35)
                draw.ellipse([(lx - rad, ly - rad), (lx + rad, ly + rad)], fill=col)
    elif lesion_type == "water_soaked": # Late blight dark irregular lesions
        for (lx, ly) in [(cx - 40, cy - 30), (cx + 30, cy - 40), (cx + 20, cy + 40)]:
            draw.ellipse([(lx - 25, ly - 35), (lx + 35, ly + 25)], fill=(45, 40, 30))
            draw.ellipse([(lx - 15, ly - 20), (lx + 20, ly + 15)], fill=(30, 28, 20))
    elif lesion_type == "pustules": # Common rust
        for py in range(int(cy - h*0.3), int(cy + h*0.3), 16):
            for px in [cx - 20, cx - 5, cx + 15]:
                draw.ellipse([(px - 4, py - 8), (px + 4, py + 8)], fill=(185, 75, 20))
    elif lesion_type == "scab": # Apple scab
        for (lx, ly, r) in [(cx - 25, cy - 35, 18), (cx + 30, cy - 10, 22), (cx - 15, cy + 30, 16)]:
            draw.ellipse([(lx - r, ly - r), (lx + r, ly + r)], fill=(40, 50, 35))
            draw.ellipse([(lx - r//2, ly - r//2), (lx + r//2, ly + r//2)], fill=(25, 30, 22))

    # Mild smoothing
    return img.filter(ImageFilter.GaussianBlur(radius=0.7))

# Generate sample set
samples = {
    "sample_tomato_early_blight.png": dict(base_color=(60, 135, 50), leaf_shape="compound", lesion_type="target_rings"),
    "sample_tomato_healthy.png": dict(base_color=(45, 155, 40), leaf_shape="compound", lesion_type=None),
    "sample_potato_late_blight.png": dict(base_color=(55, 125, 45), leaf_shape="ovate", lesion_type="water_soaked"),
    "sample_corn_common_rust.png": dict(base_color=(75, 145, 50), leaf_shape="linear", lesion_type="pustules"),
    "sample_apple_scab.png": dict(base_color=(50, 130, 45), leaf_shape="ovate", lesion_type="scab"),
}

for fname, kwargs in samples.items():
    out_path = OUTPUT_DIR / fname
    img = generate_leaf(**kwargs)
    img.save(out_path)
    print(f"Generated {out_path}")

