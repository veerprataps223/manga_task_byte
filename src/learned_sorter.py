"""
Learned Sorter: Machine learning pairwise ranking model for Manga Reading Order.
Trains a pairwise classifier P(bubble_A comes before bubble_B) and sorts candidate bubbles.
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def extract_pairwise_features(b_a: Dict[str, Any], b_b: Dict[str, Any], page_w: float, page_h: float) -> List[float]:
    """
    Features comparing bubble A and bubble B.
    Target = 1 if A should be read BEFORE B.
    Target = 0 if B should be read BEFORE A.
    """
    dy = (b_b["cy"] - b_a["cy"]) / max(1.0, page_h)  # > 0 means A is higher than B
    dx = (b_b["cx"] - b_a["cx"]) / max(1.0, page_w)  # > 0 means A is to the left of B, < 0 means A is to the right
    
    # Overlap along vertical axis
    min_y2 = min(b_a["y2"], b_b["y2"])
    max_y1 = max(b_a["y1"], b_b["y1"])
    overlap_y = max(0.0, min_y2 - max_y1) / max(b_a["h"], b_b["h"], 1.0)
    
    # Overlap along horizontal axis
    min_x2 = min(b_a["x2"], b_b["x2"])
    max_x1 = max(b_a["x1"], b_b["x1"])
    overlap_x = max(0.0, min_x2 - max_x1) / max(b_a["w"], b_b["w"], 1.0)
    
    is_same_row = 1.0 if overlap_y > 0.35 else 0.0
    is_same_col = 1.0 if overlap_x > 0.35 else 0.0
    
    # Manga rule signals:
    # 1. Top before bottom: A is higher -> dy > 0
    # 2. Right before left: A is to the right of B -> b_a["cx"] > b_b["cx"] -> dx < 0
    right_before_left = 1.0 if b_a["cx"] > b_b["cx"] else -1.0
    top_before_bottom = 1.0 if b_a["cy"] < b_b["cy"] else -1.0
    
    return [
        dy, dx, abs(dy), abs(dx), overlap_y, overlap_x,
        is_same_row, is_same_col, right_before_left, top_before_bottom
    ]

class LearnedMangaSorter:
    def __init__(self):
        self.clf = LogisticRegression(C=1.0, random_state=42)
        self.is_trained = False
        
    def train(self, X_pairs: np.ndarray, y_pairs: np.ndarray):
        self.clf.fit(X_pairs, y_pairs)
        self.is_trained = True
        acc = self.clf.score(X_pairs, y_pairs)
        print(f"LearnedMangaSorter trained on {len(X_pairs)} pairs. Pairwise accuracy: {acc:.4f}")
        print(f"Coefficients: {self.clf.coef_[0]}")
        
    def sort(self, bubbles: List[Dict[str, Any]], page_w: float, page_h: float) -> List[Dict[str, Any]]:
        if len(bubbles) <= 1:
            return bubbles
            
        n = len(bubbles)
        if not self.is_trained:
            # Rule-based manga reading order fallback:
            # Top-to-bottom bands, right-to-left within band
            def fallback_key(b):
                band = b["cy"] // 170.0
                col = - (b["cx"] // 90.0)
                return (band, col, b["cy"])
            return sorted(bubbles, key=fallback_key)
            
        # Pairwise tournament / Borda count scoring
        scores = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                feat = extract_pairwise_features(bubbles[i], bubbles[j], page_w, page_h)
                prob_i_before_j = self.clf.predict_proba([feat])[0][1]
                scores[i] += prob_i_before_j
                scores[j] += (1.0 - prob_i_before_j)
                
        # Sort descending by win score
        ranked_indices = np.argsort(-scores)
        return [bubbles[idx] for idx in ranked_indices]
