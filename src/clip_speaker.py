"""
Visual Character Re-Identification Engine using OpenAI CLIP (clip-vit-base-patch32).
Extracts 512-dimensional visual embeddings of character regions and clusters them
across all 3 consecutive manga pages to maintain consistent identity tags.
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from sklearn.cluster import AgglomerativeClustering

CLIP_MODEL = None
CLIP_PROCESSOR = None

def get_clip_model():
    global CLIP_MODEL, CLIP_PROCESSOR
    if CLIP_MODEL is None:
        model_id = "openai/clip-vit-base-patch32"
        CLIP_PROCESSOR = AutoProcessor.from_pretrained(model_id)
        CLIP_MODEL = CLIPVisionModelWithProjection.from_pretrained(model_id).cuda()
        CLIP_MODEL.eval()
    return CLIP_PROCESSOR, CLIP_MODEL

def extract_character_crop_for_bubble(
    image: Image.Image, 
    bubble: Dict[str, Any], 
    page_w: float, 
    page_h: float,
    tail_vector: Optional[Tuple[float, float]] = None
) -> Image.Image:
    """
    Extract the visual character crop associated with a speech bubble.
    If tail_vector is provided from SAM 2, shift the crop along the bubble tail pointing vector.
    Otherwise, take a contextual window downwards and sideways from the bubble coordinates.
    """
    bcx, bcy = bubble["cx"], bubble["cy"]
    
    if tail_vector is not None:
        shift_x = tail_vector[0] * 90.0
        shift_y = max(30.0, tail_vector[1] * 120.0)
        target_cx = bcx + shift_x
        target_cy = bcy + shift_y
        crop_x1 = max(0.0, target_cx - 100.0)
        crop_x2 = min(page_w, target_cx + 100.0)
        crop_y1 = max(0.0, target_cy - 80.0)
        crop_y2 = min(page_h, target_cy + 150.0)
    else:
        bw, bh = bubble["w"], bubble["h"]
        crop_x1 = max(0.0, bcx - max(bw * 1.2, 120.0))
        crop_x2 = min(page_w, bcx + max(bw * 1.2, 120.0))
        crop_y1 = max(0.0, bubble["y1"] - 30.0)
        crop_y2 = min(page_h, bubble["y2"] + max(bh * 2.5, 220.0))
    
    # Ensure minimum valid crop size
    if (crop_x2 - crop_x1) < 40 or (crop_y2 - crop_y1) < 40:
        crop_x1, crop_x2 = 0, page_w
        crop_y1, crop_y2 = 0, page_h
        
    return image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))

def attribute_sequence_with_clip(
    image_paths: List[str], 
    pages_bubbles: List[List[Dict[str, Any]]], 
    page_w: float = 900.0, 
    page_h: float = 1300.0,
    use_sam2: bool = True
) -> List[List[Dict[str, str]]]:
    """
    Cluster dialogue lines across 3 pages using CLIP visual embeddings,
    optionally guided by Meta SAM 2 speech balloon tail vectors.
    """
    processor, clip_model = get_clip_model()
    
    sam2_segmenter = None
    if use_sam2:
        try:
            from sam2_balloon import segment_speech_balloon
            sam2_segmenter = segment_speech_balloon
        except Exception:
            sam2_segmenter = None
    
    # 1. Collect non-narration dialogue bubbles and their visual crops
    dialogue_items = []
    crops = []
    
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    for p_idx, page in enumerate(pages_bubbles):
        img = images[p_idx]
        w, h = img.size
        for b_idx, b in enumerate(page):
            if b.get("assigned_speaker") == "NARRATION":
                continue
            
            tail_vec = None
            if sam2_segmenter is not None:
                try:
                    box = (b["x1"], b["y1"], b["x2"], b["y2"])
                    res = sam2_segmenter(img, box)
                    tail_vec = res.get("tail_vector")
                except Exception:
                    tail_vec = None
                    
            crop = extract_character_crop_for_bubble(img, b, w, h, tail_vector=tail_vec)
            crops.append(crop)
            dialogue_items.append((p_idx, b_idx, b))
            
    # If 0 or 1 dialogue line, assign trivial character1
    if len(dialogue_items) <= 1:
        for p_idx, b_idx, b in dialogue_items:
            b["assigned_speaker"] = "character1"
    else:
        # 2. Extract normalized 512-d CLIP embeddings in batch
        inputs = processor(images=crops, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = clip_model(**inputs)
            embeds = outputs.image_embeds
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            embeds_np = embeds.cpu().numpy()
            
        # 3. Agglomerative Clustering with Cosine Metric (K=2 primary characters)
        n_clusters = min(2, len(dialogue_items))
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
        labels = clusterer.fit_predict(embeds_np)
        
        # 4. Conversational Turn Smoothing: remove isolated single-token noise flips
        smoothed_labels = list(labels)
        for i in range(1, len(smoothed_labels) - 1):
            if smoothed_labels[i-1] == smoothed_labels[i+1] and smoothed_labels[i] != smoothed_labels[i-1]:
                p_prev, _, b_prev = dialogue_items[i-1]
                p_curr, _, b_curr = dialogue_items[i]
                if p_prev == p_curr and abs(b_curr["cy"] - b_prev["cy"]) < 90.0:
                    smoothed_labels[i] = smoothed_labels[i-1]
                    
        # 5. Map cluster labels consistently (Character 1 for most frequent / earliest speaker)
        first_label = smoothed_labels[0]
        label_map = {first_label: "character1"}
        other_labels = [l for l in smoothed_labels if l != first_label]
        if other_labels:
            label_map[other_labels[0]] = "character2"
            
        for i, (p_idx, b_idx, b) in enumerate(dialogue_items):
            cluster_id = smoothed_labels[i]
            b["assigned_speaker"] = label_map.get(cluster_id, "character1")
            
    # 5. Build final output pages
    final_pages = []
    for page in pages_bubbles:
        page_lines = []
        for b in page:
            spk = b.get("assigned_speaker", "character1")
            page_lines.append({"speaker": spk, "text": b["text"]})
        final_pages.append(page_lines)
        
    return final_pages
