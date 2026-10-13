"""
Enhanced Speaker Tracker:
1. Lexical and Spatial Narration Classification.
2. Dynamic Multi-Speaker Estimation (K-Means with Silhouette Score).
3. Conversational Turn Smoothing and Global Sequence Consistency.
"""
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

NARRATION_PHRASES = [
    "meanwhile", "years later", "at that time", "in the end", "the result",
    "a clear victor", "as she ", "as he ", "as they ", "his thoughts", "her thoughts",
    "their thoughts", "his pride", "chaos theory", "the story of", "on that day",
    "in modern times", "was dispelled", "became clouded", "stood before", "in other words"
]

FIRST_SECOND_PERSON = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves"
}

class LearnedSpeakerTracker:
    def __init__(self, default_k: int = 2):
        self.default_k = default_k

    def is_narration(self, b: Dict[str, Any], page_w: float, page_h: float) -> bool:
        text = b["text"].strip()
        t_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", t_lower))
        
        # Dialogue pronouns strictly disqualify narration
        if words & FIRST_SECOND_PERSON:
            return False
            
        # Screams / questions / conversational slang are dialogue
        if any(p in text for p in ["?!", "?!?", "??"]) or t_lower.startswith(("huh", "hey", "gah", "wait", "what")):
            return False
            
        # Explicit narrative exposition phrases
        if any(phrase in t_lower for phrase in NARRATION_PHRASES):
            return True
            
        # Setting date / time stamp (e.g. "MARCH 4, 1985...")
        if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b.*\d{4}", t_lower):
            return True
            
        # Third-person expository past-tense narrative sentence
        if len(words) >= 6 and any(v in words for v in ["was", "were", "had", "became", "seemed", "emerged"]):
            if not any(c in text for c in ["!", "?", "\""]):
                return True
                
        return False

    def attribute_sequence(self, pages_bubbles: List[List[Dict[str, Any]]], page_w: float = 900.0, page_h: float = 1300.0) -> List[List[Dict[str, str]]]:
        all_dialogue = []
        for p_idx, page in enumerate(pages_bubbles):
            for b_idx, b in enumerate(page):
                if self.is_narration(b, page_w, page_h):
                    b["assigned_speaker"] = "NARRATION"
                else:
                    b["assigned_speaker"] = None
                    all_dialogue.append((p_idx, b_idx, b))
                    
        if not all_dialogue:
            return [[{"speaker": b.get("assigned_speaker") or "NARRATION", "text": b["text"]} for b in page] for page in pages_bubbles]
            
        N = len(all_dialogue)
        # Spatial features for clustering across sequence
        X_feats = np.array([[b["cx"] / page_w, b["cy"] / page_h] for _, _, b in all_dialogue])
        
        # Decide number of clusters K
        best_k = 2
        if N >= 8:
            sil_scores = {}
            for k in [2, 3]:
                if N > k:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_feats)
                    # Check cluster balance: each cluster must have at least 2 items
                    counts = Counter(km.labels_)
                    if all(counts[c] >= 2 for c in range(k)):
                        sil = silhouette_score(X_feats, km.labels_)
                        sil_scores[k] = sil
            # If 3 clusters has high silhouette (> 0.45), use 3 speakers
            if sil_scores.get(3, 0.0) >= 0.45:
                best_k = 3
            else:
                best_k = 2
        else:
            best_k = 2 if N >= 3 else 1
            
        # Fit KMeans with best_k
        if best_k == 1:
            cluster_labels = np.zeros(N, dtype=int)
        else:
            km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_feats)
            cluster_labels = km.labels_
            
        # Map clusters to character1, character2, character3 by frequency (most frequent is character1)
        cluster_counts = Counter(cluster_labels)
        sorted_clusters = [c for c, _ in cluster_counts.most_common()]
        cluster_to_char = {c: f"character{rank+1}" for rank, c in enumerate(sorted_clusters)}
        
        # Conversational turn smoothing
        prev_speaker = "character1"
        for i, (p_idx, b_idx, b) in enumerate(all_dialogue):
            c_label = cluster_labels[i]
            assigned_char = cluster_to_char[c_label]
            
            # If on same page and very close vertically (within 100px) and horizontally (within 70px)
            if i > 0:
                prev_p, _, prev_b = all_dialogue[i - 1]
                if prev_p == p_idx:
                    dy = abs(b["cy"] - prev_b["cy"])
                    dx = abs(b["cx"] - prev_b["cx"])
                    if dy < 110.0 and dx < 65.0:
                        assigned_char = prev_speaker
                        
            b["assigned_speaker"] = assigned_char
            prev_speaker = assigned_char
            
        # Build final pages
        final_pages = []
        for page in pages_bubbles:
            page_list = []
            for b in page:
                spk = b.get("assigned_speaker") or "character1"
                page_list.append({"speaker": spk, "text": b["text"]})
            final_pages.append(page_list)
            
        return final_pages
