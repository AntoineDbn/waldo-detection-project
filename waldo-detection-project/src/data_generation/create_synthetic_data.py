"""
Synthetic Waldo Data Generator
===============================
Creates synthetic training data by pasting Waldo on clean backgrounds with augmentations.

Features:
- Random scaling, rotation, and brightness adjustments
- Automatic YOLO label generation
- Verification images with drawn bboxes
"""

import os
import argparse
import random
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np


class SyntheticDataGenerator:
    """Generate synthetic Waldo training data with augmentations."""
    
    def __init__(
        self,
        background_dir: str,
        waldo_dir: str,
        output_img_dir: str,
        output_label_dir: str,
        output_debug_dir: str = None,
        scale_range: Tuple[float, float] = (0.3, 0.8),
        rotation_range: Tuple[float, float] = (-15, 15),
        brightness_range: Tuple[float, float] = (0.7, 1.2),
        class_id: int = 0
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            background_dir: Directory with background images (no Waldo)
            waldo_dir: Directory with Waldo PNG cutouts (transparent background)
            output_img_dir: Where to save generated images
            output_label_dir: Where to save YOLO .txt labels
            output_debug_dir: Optional debug directory with drawn bboxes
            scale_range: (min, max) scale factors for Waldo
            rotation_range: (min, max) rotation in degrees
            brightness_range: (min, max) brightness multiplier
            class_id: YOLO class ID for Waldo (default: 0)
        """
        self.bg_dir = Path(background_dir)
        self.waldo_dir = Path(waldo_dir)
        self.out_img = Path(output_img_dir)
        self.out_label = Path(output_label_dir)
        self.out_debug = Path(output_debug_dir) if output_debug_dir else None
        
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.class_id = class_id
        
        # Create output directories
        self.out_img.mkdir(parents=True, exist_ok=True)
        self.out_label.mkdir(parents=True, exist_ok=True)
        if self.out_debug:
            self.out_debug.mkdir(parents=True, exist_ok=True)
        
        # Load file lists
        self.bg_paths = self._get_images(self.bg_dir)
        self.waldo_paths = self._get_images(self.waldo_dir)
        
        if not self.bg_paths:
            raise RuntimeError(f"No backgrounds found in {self.bg_dir}")
        if not self.waldo_paths:
            raise RuntimeError(f"No Waldo cutouts found in {self.waldo_dir}")
        
        print(f"📁 Loaded {len(self.bg_paths)} backgrounds")
        print(f"🎨 Loaded {len(self.waldo_paths)} Waldo cutouts\n")
    
    @staticmethod
    def _get_images(directory: Path):
        """Get all image files from directory."""
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for pattern in patterns:
            images.extend(directory.glob(pattern))
        return sorted(images)
    
    def random_transform(self, waldo_img: Image.Image) -> Image.Image:
        """
        Apply random transformations to Waldo.
        
        Args:
            waldo_img: PIL Image with alpha channel
            
        Returns:
            Transformed PIL Image
        """
        # 1. Random scale
        scale = random.uniform(*self.scale_range)
        new_w = int(waldo_img.width * scale)
        new_h = int(waldo_img.height * scale)
        img = waldo_img.resize((new_w, new_h), Image.LANCZOS)
        
        # 2. Random rotation
        angle = random.uniform(*self.rotation_range)
        img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
        
        # 3. Random brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(*self.brightness_range))
        
        return img
    
    def paste_waldo(
        self,
        background: Image.Image,
        waldo: Image.Image
    ) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
        """
        Paste Waldo on background at random position.
        
        Args:
            background: Background PIL Image (RGB)
            waldo: Waldo PIL Image (RGBA)
            
        Returns:
            (composite_image, yolo_bbox) where bbox is (x_center, y_center, width, height)
            normalized to [0, 1]
        """
        bg_w, bg_h = background.size
        w_w, w_h = waldo.size
        
        # Random position (ensure Waldo fits completely)
        if bg_w < w_w or bg_h < w_h:
            # Waldo is larger than background, resize Waldo
            scale = min(bg_w / w_w, bg_h / w_h) * 0.9
            new_w = int(w_w * scale)
            new_h = int(w_h * scale)
            waldo = waldo.resize((new_w, new_h), Image.LANCZOS)
            w_w, w_h = new_w, new_h
        
        x0 = random.randint(0, max(0, bg_w - w_w))
        y0 = random.randint(0, max(0, bg_h - w_h))
        
        # Paste with alpha mask
        composite = background.copy()
        composite.paste(waldo, (x0, y0), mask=waldo)
        
        # Compute YOLO bbox (normalized)
        x_center = (x0 + w_w / 2) / bg_w
        y_center = (y0 + w_h / 2) / bg_h
        width = w_w / bg_w
        height = w_h / bg_h
        
        return composite, (x_center, y_center, width, height)
    
    def draw_debug(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float]
    ) -> Image.Image:
        """
        Draw bounding box on image for debugging.
        
        Args:
            image: PIL Image
            bbox: YOLO format bbox (x_center, y_center, width, height) normalized
            
        Returns:
            Image with drawn bbox
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        w, h = img.size
        xc, yc, bw, bh = bbox
        
        # Convert to pixel coordinates
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        
        # Draw label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((x1, y1-22), "Waldo", fill="lime", font=font)
        
        return img
    
    def generate(self, n_per_background: int = 5):
        """
        Generate synthetic dataset.
        
        Args:
            n_per_background: Number of augmented images per background
        """
        count = 0
        
        for bg_path in self.bg_paths:
            # Load background
            background = Image.open(bg_path).convert("RGB")
            
            for i in range(n_per_background):
                # Choose random Waldo
                waldo_path = random.choice(self.waldo_paths)
                waldo = Image.open(waldo_path).convert("RGBA")
                
                # Apply transformations
                waldo_transformed = self.random_transform(waldo)
                
                # Paste on background
                composite, bbox = self.paste_waldo(background, waldo_transformed)
                
                # Generate filename
                base_name = f"{bg_path.stem}_{i:02d}"
                img_name = f"{base_name}.jpg"
                label_name = f"{base_name}.txt"
                
                # Save image
                img_path = self.out_img / img_name
                composite.save(img_path, quality=90)
                
                # Save YOLO label
                label_path = self.out_label / label_name
                xc, yc, w, h = bbox
                with open(label_path, 'w') as f:
                    f.write(f"{self.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                
                # Save debug visualization if requested
                if self.out_debug:
                    debug_img = self.draw_debug(composite, bbox)
                    debug_path = self.out_debug / f"{base_name}_debug.jpg"
                    debug_img.save(debug_path, quality=90)
                
                count += 1
                if count % 10 == 0:
                    print(f"✅ Generated {count} images...")
        
        print(f"\n🎉 Generation complete: {count} synthetic images created")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Waldo training data"
    )
    parser.add_argument(
        "--backgrounds",
        type=str,
        required=True,
        help="Directory with background images (no Waldo)"
    )
    parser.add_argument(
        "--waldo-refs",
        type=str,
        required=True,
        help="Directory with Waldo PNG cutouts (transparent)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for synthetic dataset"
    )
    parser.add_argument(
        "--n-per-bg",
        type=int,
        default=5,
        help="Number of augmented images per background (default: 5)"
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.3,
        help="Minimum scale factor (default: 0.3)"
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=0.8,
        help="Maximum scale factor (default: 0.8)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate debug images with drawn bboxes"
    )
    
    args = parser.parse_args()
    
    # Setup directories
    output_base = Path(args.output)
    img_dir = output_base / "images"
    label_dir = output_base / "labels"
    debug_dir = output_base / "debug" if args.debug else None
    
    # Create generator
    generator = SyntheticDataGenerator(
        background_dir=args.backgrounds,
        waldo_dir=args.waldo_refs,
        output_img_dir=str(img_dir),
        output_label_dir=str(label_dir),
        output_debug_dir=str(debug_dir) if debug_dir else None,
        scale_range=(args.scale_min, args.scale_max)
    )
    
    # Generate dataset
    generator.generate(n_per_background=args.n_per_bg)


if __name__ == "__main__":
    main()
