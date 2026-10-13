"""
Learned Filter: Machine learning classifier to separate Story Dialogue from SFX and Noise.
Trains a classifier on extracted features using ground-truth matching.
"""
import re
import math
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

SFX_LEXICON = {
    "bam", "bang", "boom", "clack", "click", "crash", "creak", "drip", 
    "fwip", "fwump", "gasp", "glance", "grunt", "gulp", "huff", "pant", 
    "rustle", "screeech", "shing", "shp", "shk", "sigh", "slam", "slap", 
    "slash", "smack", "snap", "squeak", "step", "swish", "tap", "thud", 
    "thump", "whoosh", "zap", "kzt", "bzz", "gah", "chk", "tch", "fsh",
    "clink", "clatter", "giggle", "snicker", "guffaw", "whack", "smash",
    "rumble", "rattle", "clang", "screech", "whirr", "beep"
}

def extract_features(b: Dict[str, Any], page_w: float, page_h: float) -> List[float]:
    text = b["text"].strip()
    clean = re.sub(r"[^\w\s]", "", text).lower()
    words = clean.split()
    n_words = len(words)
    n_chars = len(text)
    
    cx, cy = b["cx"], b["cy"]
    w, h = b["w"], b["h"]
    
    # Spatial features
    norm_cx = cx / max(1.0, page_w)
    norm_cy = cy / max(1.0, page_h)
    norm_w = w / max(1.0, page_w)
    norm_h = h / max(1.0, page_h)
    aspect_ratio = w / max(1.0, h)
    norm_area = (w * h) / max(1.0, page_w * page_h)
    
    # Margin proximity
    dist_x_margin = min(cx, page_w - cx) / max(1.0, page_w)
    dist_y_margin = min(cy, page_h - cy) / max(1.0, page_h)
    
    # Textual features
    is_digit = 1.0 if re.match(r"^\d+$", text) else 0.0
    is_punct_only = 1.0 if re.match(r"^[\.\?!]+$", text) else 0.0
    has_punct = 1.0 if any(c in ".,!?;:'\"-" for c in text) else 0.0
    upper_ratio = sum(c.isupper() for c in text) / max(1.0, n_chars)
    vowels = sum(c in "aeiou" for c in clean)
    vowel_ratio = vowels / max(1.0, len(clean))
    
    # SFX lexicon match
    in_sfx_lexicon = 1.0 if clean in SFX_LEXICON or (words and words[0] in SFX_LEXICON) else 0.0
    
    return [
        norm_cx, norm_cy, norm_w, norm_h, aspect_ratio, norm_area,
        dist_x_margin, dist_y_margin, float(n_words), float(n_chars),
        is_digit, is_punct_only, has_punct, upper_ratio, vowel_ratio, in_sfx_lexicon
    ]

class LearnedStoryFilter:
    def __init__(self):
        self.clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        self.is_trained = True
        train_acc = self.clf.score(X, y)
        print(f"LearnedStoryFilter trained on {len(X)} samples. Train accuracy: {train_acc:.4f}")
        
    def predict(self, b: Dict[str, Any], page_w: float, page_h: float) -> bool:
        if not self.is_trained:
            # Fallback to rule-based
            text = b["text"].strip()
            if not text:
                return False
            clean = re.sub(r"[^\w\s]", "", text).lower()
            if clean in SFX_LEXICON:
                return False
            if re.match(r"^\d+$", text) and (b["cy"] > 0.88 * page_h or b["cy"] < 0.12 * page_h):
                return False
            return True
            
        feat = extract_features(b, page_w, page_h)
        prob = self.clf.predict_proba([feat])[0][1]
        return prob >= 0.45
