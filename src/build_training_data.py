"""
Build training data for Learned Filter and Learned Reading Order Sorter.
Uses cached Florence OCR and gold labels to generate labeled feature sets.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from manga_processor import merge_word_boxes
from learned_filter import extract_features
from learned_sorter import extract_pairwise_features

def token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    return inter / max(len(tokens_a), len(tokens_b))

def build_training_dataset(cached_dir: str, labels_jsonl: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(labels_jsonl, "r", encoding="utf-8") as f:
        labels_map = {json.loads(line)["sequence_id"]: json.loads(line) for line in f if line.strip()}
        
    cached_path = Path(cached_dir)
    cached_files = list(cached_path.glob("*.json"))
    
    X_filter, y_filter = [], []
    X_pairs, y_pairs = [], []
    
    for cfile in cached_files:
        sid = cfile.stem
        if sid not in labels_map:
            continue
            
        with open(cfile, "r", encoding="utf-8") as f:
            seq_data = json.load(f)
            
        gold_seq = labels_map[sid]
        
        for p_idx, page_data in enumerate(seq_data["pages"]):
            if p_idx >= len(gold_seq["pages"]):
                continue
            gold_page = gold_seq["pages"][p_idx]
            w, h = page_data["width"], page_data["height"]
            
            # Merge word boxes into candidate bubbles
            bubbles = merge_word_boxes(page_data["boxes"], w, h)
            if not bubbles:
                continue
                
            # Match each bubble to gold page
            matched_bubbles = []
            for b in bubbles:
                feat = extract_features(b, w, h)
                # Find best matching gold line
                best_sim = 0.0
                best_g_idx = -1
                for g_idx, g_line in enumerate(gold_page):
                    sim = token_overlap(b["text"], g_line["text"])
                    if sim > best_sim:
                        best_sim = sim
                        best_g_idx = g_idx
                        
                is_story = (best_sim >= 0.45)
                X_filter.append(feat)
                y_filter.append(1 if is_story else 0)
                
                if is_story and best_g_idx >= 0:
                    matched_bubbles.append((b, best_g_idx))
                    
            # Generate pairwise reading order pairs
            for i in range(len(matched_bubbles)):
                for j in range(i + 1, len(matched_bubbles)):
                    b_a, order_a = matched_bubbles[i]
                    b_b, order_b = matched_bubbles[j]
                    
                    if order_a == order_b:
                        continue
                        
                    feat_ab = extract_pairwise_features(b_a, b_b, w, h)
                    # Target = 1 if a is before b in gold order
                    target_ab = 1 if order_a < order_b else 0
                    X_pairs.append(feat_ab)
                    y_pairs.append(target_ab)
                    
                    # Also include symmetric reverse pair (b, a)
                    feat_ba = extract_pairwise_features(b_b, b_a, w, h)
                    X_pairs.append(feat_ba)
                    y_pairs.append(1 - target_ab)
                    
    X_filter = np.array(X_filter) if X_filter else np.empty((0, 16))
    y_filter = np.array(y_filter) if y_filter else np.empty((0,))
    X_pairs = np.array(X_pairs) if X_pairs else np.empty((0, 10))
    y_pairs = np.array(y_pairs) if y_pairs else np.empty((0,))
    
    print(f"Generated {len(X_filter)} filter samples (positives={sum(y_filter==1)}, negatives={sum(y_filter==0)})")
    print(f"Generated {len(X_pairs)} pairwise ranking samples")
    return X_filter, y_filter, X_pairs, y_pairs

if __name__ == "__main__":
    X_f, y_f, X_p, y_p = build_training_dataset("cache/florence_ocr", "data/val_labels.jsonl")
