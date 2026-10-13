"""
Adapted Manga Processor:
Integrates learned ML components:
1. Learned Story Filter (Gradient Boosting Classifier)
2. Learned Manga Reading Order Sorter (Pairwise Logistic Regression)
3. Learned Speaker Attribution & Cross-Page Character Consistency Tracker
"""
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from learned_filter import extract_features
from learned_sorter import extract_pairwise_features
from learned_speaker import LearnedSpeakerTracker

SFX_LEXICON = {
    "bam", "bang", "boom", "clack", "click", "crash", "creak", "drip", 
    "fwip", "fwump", "gasp", "glance", "grunt", "gulp", "huff", "pant", 
    "rustle", "screeech", "shing", "shp", "shk", "sigh", "slam", "slap", 
    "slash", "smack", "snap", "squeak", "step", "swish", "tap", "thud", 
    "thump", "whoosh", "zap", "kzt", "bzz", "gah", "chk", "tch", "fsh",
    "clink", "clatter", "giggle", "snicker", "guffaw", "whack", "smash",
    "rumble", "rattle", "clang", "screech", "whirr", "beep"
}

# Global cached instances
FILTER_MODEL = None
SORTER_MODEL = None
SPEAKER_TRACKER = LearnedSpeakerTracker()

def load_models():
    global FILTER_MODEL, SORTER_MODEL
    filter_p = Path("models/story_filter.pkl")
    sorter_p = Path("models/reading_order_sorter.pkl")
    if filter_p.exists() and FILTER_MODEL is None:
        with open(filter_p, "rb") as f:
            FILTER_MODEL = pickle.load(f)
    if sorter_p.exists() and SORTER_MODEL is None:
        with open(sorter_p, "rb") as f:
            SORTER_MODEL = pickle.load(f)

def box_iou(b1: Dict[str, Any], b2: Dict[str, Any]) -> float:
    x1 = max(b1["x1"], b2["x1"])
    y1 = max(b1["y1"], b2["y1"])
    x2 = min(b1["x2"], b2["x2"])
    y2 = min(b1["y2"], b2["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    a2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def box_overlap_min(b1: Dict[str, Any], b2: Dict[str, Any]) -> float:
    x1 = max(b1["x1"], b2["x1"])
    y1 = max(b1["y1"], b2["y1"])
    x2 = min(b1["x2"], b2["x2"])
    y2 = min(b1["y2"], b2["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    a2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
    min_a = min(a1, a2)
    return inter / min_a if min_a > 0 else 0.0

def deduplicate_boxes(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Spatial Non-Maximum Suppression (NMS) for overlapping Florence-2 predictions."""
    deduped = []
    for b in boxes:
        is_dup = False
        for existing in deduped:
            iou = box_iou(b, existing)
            overlap = box_overlap_min(b, existing)
            if iou > 0.45 or overlap > 0.65:
                is_dup = True
                break
        if not is_dup:
            deduped.append(b)
    return deduped

def merge_word_boxes(boxes: List[Dict[str, Any]], page_w: float, page_h: float) -> List[Dict[str, Any]]:
    load_models()
    
    # Pre-filter obvious margin junk and page numbers
    valid = []
    for b in boxes:
        text = b["text"].strip()
        if not text:
            continue
        cx, cy = b["cx"], b["cy"]
        if re.match(r"^\d{1,3}$", text) and (cy > 0.88 * page_h or cy < 0.12 * page_h):
            continue
        if cx < 0.035 * page_w or cx > 0.965 * page_w:
            continue
        if cy < 0.025 * page_h or cy > 0.975 * page_h:
            continue
        valid.append(b)
        
    if not valid:
        return []
        
    # Spatial NMS: deduplicate multi-candidate Florence-2 bounding boxes
    valid = deduplicate_boxes(valid)
    valid.sort(key=lambda b: (b["cy"], b["cx"]))
    
    bubbles = []
    used = [False] * len(valid)
    
    for i in range(len(valid)):
        if used[i]:
            continue
        cluster = [valid[i]]
        used[i] = True
        
        changed = True
        while changed:
            changed = False
            for j in range(len(valid)):
                if used[j]:
                    continue
                cand = valid[j]
                for member in cluster:
                    dy = abs(cand["cy"] - member["cy"])
                    dx = abs(cand["cx"] - member["cx"])
                    h_line = max(member["y2"] - member["y1"], cand["y2"] - cand["y1"], 15.0)
                    w_box = max(member["x2"] - member["x1"], cand["x2"] - cand["x1"], 25.0)
                    
                    if dy <= 1.6 * h_line and dx <= max(w_box * 0.75, 42.0):
                        cluster.append(cand)
                        used[j] = True
                        changed = True
                        break
                    if dy <= 0.5 * h_line and abs(cand["x1"] - member["x2"]) <= 32.0:
                        cluster.append(cand)
                        used[j] = True
                        changed = True
                        break
                        
        cluster.sort(key=lambda b: (b["cy"] // 14.0, b["cx"]))
        bubble_text = " ".join(b["text"] for b in cluster)
        bubble_text = re.sub(r"(\b[A-Za-z]+)-\s+([A-Za-z]+\b)", r"\1\2", bubble_text)
        bubble_text = re.sub(r"\s+", " ", bubble_text).strip()
        
        min_x = min(b["x1"] for b in cluster)
        max_x = max(b["x2"] for b in cluster)
        min_y = min(b["y1"] for b in cluster)
        max_y = max(b["y2"] for b in cluster)
        
        bubble_obj = {
            "text": bubble_text,
            "x1": min_x,
            "y1": min_y,
            "x2": max_x,
            "y2": max_y,
            "cx": (min_x + max_x) / 2.0,
            "cy": (min_y + max_y) / 2.0,
            "w": max_x - min_x,
            "h": max_y - min_y
        }
        
        # Filter out non-story elements: CJK watermarks, SFX captions, scanlator footnotes
        if re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", bubble_text):
            continue
        if re.search(r"^(sfx|sfx\.|sfx:|tl|tl:|t/n|t/n:|tin:)", bubble_text, re.IGNORECASE):
            continue
        if re.search(r"(scanlation|reference of|translator|hidden meaning)", bubble_text, re.IGNORECASE):
            continue
        clean_lower = re.sub(r"[^\w\s]", "", bubble_text).lower().strip()
        if clean_lower in {"sfx", "vroom", "whoosh", "sizzle", "rush", "grip", "flap", "neigh"}:
            continue
        if (min_y < 0.035 * page_h or max_y > 0.965 * page_h) and len(bubble_text.split()) <= 2:
            continue

        if FILTER_MODEL is not None:
            feat = extract_features(bubble_obj, page_w, page_h)
            prob_story = FILTER_MODEL.predict_proba([feat])[0][1]
            if prob_story < 0.40:
                continue
        else:
            if clean_lower in SFX_LEXICON:
                continue
                
        bubbles.append(bubble_obj)
        
    return bubbles

def sort_manga_order(bubbles: List[Dict[str, Any]], page_w: float = 900.0, page_h: float = 1300.0) -> List[Dict[str, Any]]:
    if len(bubbles) <= 1:
        return bubbles
        
    load_models()
    
    if SORTER_MODEL is not None:
        n = len(bubbles)
        scores = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                feat = extract_pairwise_features(bubbles[i], bubbles[j], page_w, page_h)
                prob_i_before_j = SORTER_MODEL.predict_proba([feat])[0][1]
                scores[i] += prob_i_before_j
                scores[j] += (1.0 - prob_i_before_j)
        ranked_indices = np.argsort(-scores)
        return [bubbles[idx] for idx in ranked_indices]
    else:
        sorted_by_y = sorted(bubbles, key=lambda b: b["cy"])
        bands = []
        curr_band = [sorted_by_y[0]]
        for i in range(1, len(sorted_by_y)):
            prev = curr_band[-1]
            curr = sorted_by_y[i]
            if curr["cy"] - prev["cy"] > 170.0:
                bands.append(curr_band)
                curr_band = [curr]
            else:
                curr_band.append(curr)
        bands.append(curr_band)
        
        ordered = []
        for band in bands:
            band_sorted = sorted(band, key=lambda b: (- (b["cx"] // 90.0), b["cy"]))
            ordered.extend(band_sorted)
        return ordered

def attribute_speakers_for_sequence(pages_bubbles: List[List[Dict[str, Any]]], page_w: float = 900.0, page_h: float = 1300.0) -> List[List[Dict[str, str]]]:
    return SPEAKER_TRACKER.attribute_sequence(pages_bubbles, page_w, page_h)
