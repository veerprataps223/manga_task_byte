"""
Baseline 1: Modular OCR Pipeline for 3-page Manga Text Extraction & Speaker Attribution
Uses EasyOCR for text detection/recognition, spatial heuristics for manga reading order,
and spatial/turn-taking clustering for speaker attribution.
"""
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import easyocr
import numpy as np
from PIL import Image

# Common SFX tokens to filter out
COMMON_SFX = {
    "bam", "bang", "boom", "clack", "click", "crash", "creak", "drip", 
    "fwip", "gasp", "glance", "grunt", "gulp", "huff", "pant", "rustle", 
    "screeech", "shing", "sigh", "slam", "slap", "slash", "smack", "snap", 
    "squeak", "step", "swish", "tap", "thud", "thump", "whoosh", "zap"
}

def is_sfx_or_noise(text: str, conf: float) -> bool:
    cleaned = re.sub(r"[^\w\s]", "", text).strip().lower()
    if not cleaned:
        # Punctuation-only like ..., ?!, ! should be kept if confidence is decent
        if re.match(r"^[\.\?!]+$", text.strip()) and conf > 0.3:
            return False
        return True
    if cleaned in COMMON_SFX:
        return True
    if len(cleaned) == 1 and conf < 0.6:
        return True
    return False

def merge_text_boxes(boxes: List[Dict[str, Any]], x_thresh: float = 60.0, y_thresh: float = 35.0) -> List[Dict[str, Any]]:
    """
    Merge text lines that belong to the same speech bubble.
    Two boxes are merged if they are vertically close and horizontally aligned.
    """
    if not boxes:
        return []
    
    # Sort boxes roughly by top y
    boxes = sorted(boxes, key=lambda b: (b["cy"], -b["cx"]))
    merged = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]:
            continue
        curr = boxes[i]
        curr_lines = [curr]
        used[i] = True
        
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                cand = boxes[j]
                # Check distance to any line in curr_lines
                for line in curr_lines:
                    dy = cand["y1"] - line["y2"]
                    dx = abs(cand["cx"] - line["cx"])
                    # Cand is just below line and aligned horizontally
                    if 0 <= dy <= y_thresh and dx <= x_thresh:
                        curr_lines.append(cand)
                        used[j] = True
                        changed = True
                        break
        
        # Merge curr_lines
        curr_lines.sort(key=lambda b: b["y1"]) # top to bottom within bubble
        combined_text = " ".join(b["text"] for b in curr_lines)
        min_x1 = min(b["x1"] for b in curr_lines)
        min_y1 = min(b["y1"] for b in curr_lines)
        max_x2 = max(b["x2"] for b in curr_lines)
        max_y2 = max(b["y2"] for b in curr_lines)
        avg_conf = sum(b["conf"] for b in curr_lines) / len(curr_lines)
        
        merged.append({
            "text": combined_text,
            "x1": min_x1,
            "y1": min_y1,
            "x2": max_x2,
            "y2": max_y2,
            "cx": (min_x1 + max_x2) / 2.0,
            "cy": (min_y1 + max_y2) / 2.0,
            "conf": avg_conf
        })
        
    return merged

def manga_reading_order_sort(boxes: List[Dict[str, Any]], page_width: float, page_height: float) -> List[Dict[str, Any]]:
    """
    Sort text boxes according to standard Manga Reading Order:
    Manga is read Right-to-Left, Top-to-Bottom.
    We approximate panel structure by sorting primarily by vertical band / panel rows,
    and within each band from right to left.
    """
    if not boxes:
        return []
    
    # Simple band-based sorting:
    # Divide page into dynamic vertical bands or score:
    # In manga, moving down increases y, moving left decreases x (which is later in order).
    # Sorting key: band_y = cy // (page_height / 4), then -cx
    def sort_key(b):
        band = b["cy"] // 250.0  # approximate panel height band
        return (band, -b["cx"])
    
    return sorted(boxes, key=sort_key)

class BaselineOCRPipeline:
    def __init__(self, gpu: bool = True):
        self.reader = easyocr.Reader(["en"], gpu=gpu)
        
    def process_page(self, img_path: str) -> List[Dict[str, Any]]:
        im = Image.open(img_path)
        w, h = im.size
        
        raw_results = self.reader.readtext(img_path)
        valid_boxes = []
        for bbox, text, conf in raw_results:
            if not is_sfx_or_noise(text, conf):
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                valid_boxes.append({
                    "text": text.strip(),
                    "x1": min(xs),
                    "y1": min(ys),
                    "x2": max(xs),
                    "y2": max(ys),
                    "cx": sum(xs) / 4.0,
                    "cy": sum(ys) / 4.0,
                    "conf": conf
                })
                
        # Merge vertically adjacent lines in the same bubble
        merged_bubbles = merge_text_boxes(valid_boxes)
        
        # Sort in manga reading order (RTL, Top-to-Bottom)
        ordered_bubbles = manga_reading_order_sort(merged_bubbles, w, h)
        return ordered_bubbles

    def process_sequence(self, image_paths: List[str]) -> List[List[Dict[str, str]]]:
        """
        Process 3 consecutive pages and attribute speakers across all pages.
        """
        sequence_pages_bubbles = []
        all_cx = []
        for p in image_paths:
            bubbles = self.process_page(p)
            sequence_pages_bubbles.append(bubbles)
            for b in bubbles:
                all_cx.append(b["cx"])
                
        # Simple spatial speaker attribution:
        # Cluster cx positions across the 3 pages into 2 main character columns (e.g. left speaker vs right speaker)
        # Plus detect NARRATION if bubble is at top edge or wide rectangle
        if all_cx:
            med_x = float(np.median(all_cx))
        else:
            med_x = 500.0
            
        final_pages = []
        prev_speaker = "character1"
        for page_bubbles in sequence_pages_bubbles:
            page_output = []
            for b in page_bubbles:
                text = b["text"]
                # Heuristic for narration: at extreme top or bottom border and wide aspect ratio
                width = b["x2"] - b["x1"]
                height = max(1.0, b["y2"] - b["y1"])
                if (b["y1"] < 100 or b["y2"] > 1150) and (width / height > 3.0):
                    speaker = "NARRATION"
                else:
                    # Dialogue: assign based on left vs right position
                    if b["cx"] > med_x:
                        speaker = "character1"
                    else:
                        speaker = "character2"
                page_output.append({"speaker": speaker, "text": text})
            final_pages.append(page_output)
            
        return final_pages

def run_baseline_evaluation(val_sequences_path: str, data_dir: str, output_path: str):
    with open(val_sequences_path, "r", encoding="utf-8") as f:
        val_seqs = json.load(f)
        
    pipeline = BaselineOCRPipeline(gpu=True)
    results = []
    
    print(f"Running Baseline 1 (OCR) on {len(val_seqs)} validation sequences...")
    for i, seq in enumerate(val_seqs):
        seq_id = seq["sequence_id"]
        img_paths = [str(Path(data_dir) / img) for img in seq["images"]]
        pages_pred = pipeline.process_sequence(img_paths)
        results.append({
            "sequence_id": seq_id,
            "pages": pages_pred
        })
        print(f"[{i+1}/{len(val_seqs)}] Processed {seq_id}")
        
    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"Baseline predictions saved to {output_path}")

if __name__ == "__main__":
    run_baseline_evaluation(
        val_sequences_path="data/val_sequences.json",
        data_dir="BYTE ML Task Data",
        output_path="predictions_baseline_ocr.jsonl"
    )
