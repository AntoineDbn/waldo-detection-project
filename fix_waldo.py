import os

# Le contenu CORRIGÉ du fichier detect_with_clip.py
new_content = r'''"""
Waldo Detection with YOLO + CLIP Re-ranking
============================================
Main inference pipeline for detecting Waldo in large images.
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor


class WaldoDetector:
    """Complete Waldo detection pipeline with YOLO + CLIP."""
    
    def __init__(
        self,
        yolo_model_path: str,
        prototype_dir: str,
        yolo_conf: float = 0.5,
        clip_threshold: float = 0.3,
        tile_size: int = 640,
        overlap: int = 100,
        iou_thresh: float = 0.4,
        topk: int = 5,
        min_conf: float = 0.5,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load YOLO
        print(f"📦 Loading YOLO model from {yolo_model_path}...")
        self.yolo = YOLO(yolo_model_path)
        self.yolo_conf = yolo_conf
        
        # Load CLIP
        print(f"🔄 Loading CLIP model on {self.device}...")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model.eval()
        
        # Create CLIP prototypes
        self.prototypes = self._load_prototypes(prototype_dir)
        print(f"✅ Loaded {len(self.prototypes)} CLIP prototypes\n")
        
        # Detection parameters
        self.clip_threshold = clip_threshold
        self.tile_size = tile_size
        self.overlap = overlap
        self.iou_thresh = iou_thresh
        self.topk = topk
        self.min_conf = min_conf
    
    def _load_prototypes(self, prototype_dir: str) -> List[torch.Tensor]:
        """Load and encode reference Waldo images."""
        prototypes = []
        
        if not os.path.exists(prototype_dir):
            raise FileNotFoundError(f"Reference directory not found: {prototype_dir}")

        for fname in os.listdir(prototype_dir):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            img_path = os.path.join(prototype_dir, fname)
            img = Image.open(img_path).convert("RGB")
            inputs = self.clip_processor(images=img, return_tensors="pt")
            
            # --- FIX: Utilisation de get_image_features ---
            with torch.no_grad():
                emb = self.clip_model.get_image_features(**inputs.to(self.device))
                emb = emb / emb.norm(dim=-1, keepdim=True)
            # ----------------------------------------------
            
            prototypes.append(emb)
        
        if not prototypes:
            raise RuntimeError(f"No prototype images found in {prototype_dir}")
        
        return prototypes
    
    def clip_rerank(self, crop: np.ndarray) -> float:
        """Compute CLIP similarity between crop and Waldo prototypes."""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        inputs = self.clip_processor(images=img_pil, return_tensors="pt")
        
        # --- FIX: Utilisation de get_image_features ---
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs.to(self.device))
            emb = emb / emb.norm(dim=-1, keepdim=True)
        # ----------------------------------------------
        
        # Compute similarity with each prototype
        similarities = [
            torch.cosine_similarity(emb, proto, dim=-1).item()
            for proto in self.prototypes
        ]
        return max(similarities)
    
    @staticmethod
    def compute_iou(box1: Tuple, box2: Tuple) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def merge_detections(self, detections: List[Tuple]) -> List[Tuple]:
        if not detections:
            return []
        
        merged = []
        used = set()
        
        for i, (x1, y1, x2, y2, score, cls) in enumerate(detections):
            if i in used:
                continue
            
            bx1, by1, bx2, by2 = x1, y1, x2, y2
            sum_score = score
            used.add(i)
            
            for j in range(i + 1, len(detections)):
                if j in used: continue
                
                x1b, y1b, x2b, y2b, scoreb, clsb = detections[j]
                if clsb != cls: continue
                
                iou = self.compute_iou((bx1, by1, bx2, by2), (x1b, y1b, x2b, y2b))
                
                if iou > self.iou_thresh:
                    bx1 = min(bx1, x1b)
                    by1 = min(by1, y1b)
                    bx2 = max(bx2, x2b)
                    by2 = max(by2, y2b)
                    sum_score += scoreb
                    used.add(j)
            
            merged.append((bx1, by1, bx2, by2, sum_score, cls))
        
        merged = [m for m in merged if m[4] >= self.min_conf]
        merged = sorted(merged, key=lambda x: x[4], reverse=True)
        return merged[:self.topk]
    
    def tile_and_detect(self, img: np.ndarray) -> List[Tuple]:
        h, w = img.shape[:2]
        stride = self.tile_size - self.overlap
        candidates = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1 = y
                x1 = x
                y2 = min(y1 + self.tile_size, h)
                x2 = min(x1 + self.tile_size, w)
                
                y1 = max(0, y2 - self.tile_size)
                x1 = max(0, x2 - self.tile_size)
                
                tile = img[y1:y2, x1:x2]
                results = self.yolo(tile, conf=self.yolo_conf, verbose=False)[0]
                
                for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                    xA, yA, xB, yB = map(int, box.tolist())
                    candidates.append((xA + x1, yA + y1, xB + x1, yB + y1, float(score), int(cls)))
        
        return self.merge_detections(candidates)
    
    def post_filter_clip(self, candidates: List[Tuple], img: np.ndarray) -> List[Tuple]:
        if not candidates:
            return []
        
        best = None
        best_sim = self.clip_threshold
        
        for x1, y1, x2, y2, score, cls in candidates:
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1: continue

            crop = img[y1:y2, x1:x2]
            sim = self.clip_rerank(crop)
            
            if sim > best_sim:
                best_sim = sim
                best = (x1, y1, x2, y2, sim, cls)
        
        return [best] if best is not None else []
    
    def draw_detections(self, img: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        output = img.copy()
        for x1, y1, x2, y2, score, cls in detections:
            color = (0, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"Waldo: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(output, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return output
    
    def process_folder(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for fname in sorted(os.listdir(input_dir)):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)
            
            img = cv2.imread(input_path)
            if img is None: continue
                
            detections = self.tile_and_detect(img)
            final = self.post_filter_clip(detections, img)
            
            output_img = self.draw_detections(img, final)
            cv2.imwrite(output_path, output_img)
            
            if final: print(f"✅ {fname}: Waldo detected (CLIP score = {final[0][4]:.2f})")
            else: print(f"❌ {fname}: Waldo not detected")
        print(f"\n🎉 Processing complete! Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Detect Waldo using YOLO + CLIP pipeline")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--refs", type=str, required=True)
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--clip-threshold", type=float, default=0.3)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    detector = WaldoDetector(
        yolo_model_path=args.model,
        prototype_dir=args.refs,
        yolo_conf=args.yolo_conf,
        clip_threshold=args.clip_threshold,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device
    )
    
    if os.path.isdir(args.input):
        detector.process_folder(args.input, args.output)
    else:
        os.makedirs(args.output, exist_ok=True)
        img = cv2.imread(args.input)
        if img is None:
            print(f"❌ Error: Could not read image at {args.input}")
            return
            
        detections = detector.tile_and_detect(img)
        final = detector.post_filter_clip(detections, img)
        
        output_img = detector.draw_detections(img, final)
        output_path = os.path.join(args.output, os.path.basename(args.input))
        cv2.imwrite(output_path, output_img)
        
        if final:
            print(f"✅ Waldo detected! (CLIP score = {final[0][4]:.2f})")
            print(f"Saved result to {output_path}")
        else:
            print("❌ Waldo not detected")

if __name__ == "__main__":
    main()
'''

target_path = os.path.join("src", "inference", "detect_with_clip.py")
with open(target_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"✅ Fichier réparé avec succès : {target_path}")