"""
CLIP-based Waldo Extraction
============================
Extracts Waldo from annotated images using yellow circle detection and CLIP validation.

This script:
1. Detects yellow circles in annotated images
2. Uses CLIP embeddings to validate if the circled area contains Waldo
3. Saves validated Waldo crops with transparent backgrounds
"""

import os
import argparse
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class WaldoExtractor:
    """Extract Waldo instances from annotated images using CLIP validation."""
    
    def __init__(
        self,
        ref_dir: str,
        similarity_threshold: float = 0.70,
        device: str = None
    ):
        """
        Initialize the Waldo extractor.
        
        Args:
            ref_dir: Directory containing reference Waldo images
            similarity_threshold: Minimum CLIP similarity to accept detection
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.ref_dir = ref_dir
        self.sim_threshold = similarity_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # HSV range for yellow circle detection
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])
        
        # Load CLIP model and create prototype
        print(f"🔄 Loading CLIP model on {self.device}...")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        self.prototype = self._create_prototype()
        print(f"✅ CLIP prototype created from {len(self._get_ref_paths())} reference images\n")
    
    def _get_ref_paths(self):
        """Get all reference image paths."""
        patterns = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(os.path.join(self.ref_dir, pattern)))
        return sorted(paths)
    
    def _create_prototype(self):
        """Create CLIP prototype embedding from reference images."""
        ref_paths = self._get_ref_paths()
        if not ref_paths:
            raise RuntimeError(f"No reference images found in '{self.ref_dir}'")
        
        embeddings = []
        for ref_path in ref_paths:
            img = Image.open(ref_path).convert("RGB")
            inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                emb = self.clip_model.get_image_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(emb)
        
        # Average and normalize
        prototype = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
        prototype = prototype / prototype.norm(p=2, dim=-1, keepdim=True)
        return prototype
    
    def detect_circles(self, bgr_img: np.ndarray):
        """
        Detect yellow circles in the image.
        
        Args:
            bgr_img: BGR image from cv2.imread
            
        Returns:
            Array of circles [[x, y, r], ...] or None
        """
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        blur = cv2.GaussianBlur(mask, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=200
        )
        
        return None if circles is None else np.round(circles[0]).astype(int)
    
    def score_circle(self, pil_img: Image.Image, circle):
        """
        Compute CLIP similarity score for a detected circle.
        
        Args:
            pil_img: PIL Image
            circle: (x, y, r) tuple
            
        Returns:
            (similarity_score, bbox) tuple
        """
        x, y, r = circle
        w, h = pil_img.size
        
        # Extract region
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w, x + r), min(h, y + r)
        crop = pil_img.crop((x0, y0, x1, y1)).convert("RGB")
        
        # Compute CLIP embedding
        inputs = self.clip_processor(images=crop, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        
        # Similarity with prototype
        similarity = (emb @ self.prototype.T).item()
        return similarity, (x0, y0, x1, y1)
    
    def extract_waldo(self, image_path: str, output_path: str) -> bool:
        """
        Extract Waldo from a single annotated image.
        
        Args:
            image_path: Path to annotated image with yellow circle
            output_path: Path to save extracted Waldo (PNG with alpha)
            
        Returns:
            True if Waldo was successfully extracted, False otherwise
        """
        # Load image
        bgr = cv2.imread(image_path)
        pil = Image.open(image_path)
        
        # Detect circles
        circles = self.detect_circles(bgr)
        if circles is None or len(circles) == 0:
            return False
        
        # Score all circles and find best match
        scored = [self.score_circle(pil, c) for c in circles]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_sim, best_box = scored[0]
        
        # Check if similarity passes threshold
        if best_sim < self.sim_threshold:
            return False
        
        # Extract and remove yellow circle
        x0, y0, x1, y1 = best_box
        crop = pil.crop((x0, y0, x1, y1)).convert("RGBA")
        arr = np.array(crop)
        
        # Remove yellow pixels
        hsv_crop = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2HSV)
        mask_yellow = cv2.inRange(hsv_crop, self.yellow_lower, self.yellow_upper)
        arr[mask_yellow > 0, 3] = 0  # Set alpha to 0 for yellow pixels
        
        # Save
        Image.fromarray(arr).save(output_path)
        return True
    
    def process_folder(self, input_dir: str, output_dir: str):
        """
        Process all images in a folder.
        
        Args:
            input_dir: Directory with annotated images
            output_dir: Directory to save extracted Waldo crops
        """
        os.makedirs(output_dir, exist_ok=True)
        
        files = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        success_count = 0
        for filename in sorted(files):
            input_path = os.path.join(input_dir, filename)
            output_name = os.path.splitext(filename)[0] + "_waldo.png"
            output_path = os.path.join(output_dir, output_name)
            
            if self.extract_waldo(input_path, output_path):
                success_count += 1
                print(f"✅ {filename} → {output_name}")
            else:
                print(f"❌ {filename} → No valid Waldo detected")
        
        print(f"\n🎉 Extraction complete: {success_count}/{len(files)} successful")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Waldo from annotated images using CLIP validation"
    )
    parser.add_argument(
        "--annotated",
        type=str,
        required=True,
        help="Directory containing annotated images with yellow circles"
    )
    parser.add_argument(
        "--refs",
        type=str,
        required=True,
        help="Directory containing reference Waldo images"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for extracted Waldo crops"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="CLIP similarity threshold (default: 0.70)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'cpu', or None for auto"
    )
    
    args = parser.parse_args()
    
    # Create extractor and process
    extractor = WaldoExtractor(
        ref_dir=args.refs,
        similarity_threshold=args.threshold,
        device=args.device
    )
    
    extractor.process_folder(args.annotated, args.output)


if __name__ == "__main__":
    main()
