import os
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor

class WaldoDetector:
    def __init__(self, yolo_model_path, prototype_dir, yolo_conf=0.5, clip_threshold=0.3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading YOLO: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        self.yolo_conf = yolo_conf
        self.clip_threshold = clip_threshold  # <--- C'EST LA LIGNE QUI MANQUAIT !
        
        print(f"Loading CLIP on {self.device}...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        
        self.prototypes = self._load_prototypes(prototype_dir)
        print(f"Loaded {len(self.prototypes)} references.")

    def _get_safe_emb(self, inputs):
        """Fonction de securite pour extraire les embeddings peu importe la version de transformers"""
        with torch.no_grad():
            output = self.clip_model.get_image_features(**inputs.to(self.device))
            if hasattr(output, 'pooler_output'): return output.pooler_output
            if hasattr(output, 'image_embeds'): return output.image_embeds
            return output

    def _load_prototypes(self, prototype_dir):
        prototypes = []
        if not os.path.exists(prototype_dir):
            raise FileNotFoundError(f"Folder not found: {prototype_dir}")
        
        valid_exts = ('.jpg', '.png', '.jpeg')
        for fname in os.listdir(prototype_dir):
            if not fname.lower().endswith(valid_exts): continue
            
            img_path = os.path.join(prototype_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = self.clip_processor(images=img, return_tensors="pt")
                
                emb = self._get_safe_emb(inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                
                prototypes.append(emb)
            except Exception as e:
                print(f"Error on {fname}: {e}")

        if not prototypes:
            raise RuntimeError(f"No valid images in {prototype_dir}")
        return prototypes

    def get_clip_score(self, crop_img):
        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        inputs = self.clip_processor(images=img_pil, return_tensors="pt")
        
        emb = self._get_safe_emb(inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        
        scores = [torch.cosine_similarity(emb, proto, dim=-1).item() for proto in self.prototypes]
        return max(scores) if scores else 0.0

    def detect(self, image_path, output_path, tile_size=640, overlap=100):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot read image: {image_path}")
            return

        h, w = img.shape[:2]
        detections = []
        stride = tile_size - overlap
        
        print("Analyzing image tiles...")
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x1 = x
                y1 = y
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                x1 = max(0, x2 - tile_size)
                y1 = max(0, y2 - tile_size)
                
                tile = img[y1:y2, x1:x2]
                results = self.yolo(tile, conf=self.yolo_conf, verbose=False)[0]
                
                for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
                    bx1, by1, bx2, by2 = map(int, box.tolist())
                    gx1, gy1, gx2, gy2 = bx1+x1, by1+y1, bx2+x1, by2+y1
                    detections.append((gx1, gy1, gx2, gy2, float(conf)))

        best_box = None
        best_score = -1
        
        print(f"Checking {len(detections)} candidates with CLIP...")
        for x1, y1, x2, y2, conf in detections:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1: continue
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            clip_score = self.get_clip_score(crop)
            
            if clip_score > self.clip_threshold and clip_score > best_score:
                best_score = clip_score
                best_box = (x1, y1, x2, y2)

        output_img = img.copy()
        if best_box:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"Charlie! ({best_score:.2f})"
            cv2.putText(output_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"WALDO FOUND! Score: {best_score:.2f}")
        else:
            print("Waldo not found.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_img)
        print(f"Result saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--clip-threshold", type=float, default=0.3)
    args = parser.parse_args()

    detector = WaldoDetector(args.model, args.refs, args.yolo_conf, args.clip_threshold)
    
    if os.path.isdir(args.input):
        pass 
    else:
        out_file = os.path.join(args.output, os.path.basename(args.input))
        detector.detect(args.input, out_file)
