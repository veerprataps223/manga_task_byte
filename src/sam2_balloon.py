"""
Meta SAM 2 (Segment Anything Model 2) Speech Balloon & Tail Vector Segmenter.
Uses facebook/sam2-hiera-tiny on GPU to extract precise speech balloon masks
and compute directional tail pointing vectors toward speaking characters.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from transformers import Sam2Model, Sam2Processor

SAM2_PROCESSOR = None
SAM2_MODEL = None

def get_sam2():
    global SAM2_PROCESSOR, SAM2_MODEL
    if SAM2_MODEL is None:
        model_id = "facebook/sam2-hiera-tiny"
        SAM2_PROCESSOR = Sam2Processor.from_pretrained(model_id)
        SAM2_MODEL = Sam2Model.from_pretrained(model_id).cuda()
        SAM2_MODEL.eval()
    return SAM2_PROCESSOR, SAM2_MODEL

def segment_speech_balloon(
    image: Image.Image, 
    box: Tuple[float, float, float, float]
) -> Dict[str, Any]:
    """
    Prompt SAM 2 with a speech balloon bounding box [x1, y1, x2, y2].
    Returns the binary mask, balloon center, and tail tip coordinates.
    """
    processor, model = get_sam2()
    w, h = image.size
    
    input_boxes = [[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]]]
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    raw_masks = outputs.pred_masks # [1, 1, 3, 256, 256]
    best_mask = raw_masks[0, 0, 0:1]
    upsampled = F.interpolate(best_mask.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
    mask_binary = (upsampled[0, 0].cpu().numpy() > 0.0).astype(np.uint8) * 255
    
    # Analyze contour and tail vector
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "mask": mask_binary,
            "center": ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0),
            "tail_tip": None,
            "tail_vector": (0.0, 1.0)
        }
        
    largest_cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_cnt)
    if M["m00"] > 0:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        
    pts = largest_cnt[:, 0, :]
    # The tail is the point furthest downwards/sideways protruding from the box boundary
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    furthest_pt = pts[np.argmax(dists)]
    
    tail_tip = (float(furthest_pt[0]), float(furthest_pt[1]))
    tail_vec = (tail_tip[0] - cx, tail_tip[1] - cy)
    norm = max(1.0, np.sqrt(tail_vec[0]**2 + tail_vec[1]**2))
    norm_vec = (tail_vec[0] / norm, tail_vec[1] / norm)
    
    return {
        "mask": mask_binary,
        "center": (cx, cy),
        "tail_tip": tail_tip,
        "tail_vector": norm_vec,
        "contour_points": len(largest_cnt)
    }

def visualize_sam2_balloon(image_path: str, box: Tuple[float, float, float, float], out_path: str):
    """Save an annotated visual overlay of SAM 2's segmented balloon and pointing vector."""
    image = Image.open(image_path).convert("RGB")
    res = segment_speech_balloon(image, box)
    
    img_cv = cv2.imread(image_path)
    mask = res["mask"]
    
    # Draw green translucent mask
    color_mask = np.zeros_like(img_cv)
    color_mask[mask > 0] = [0, 255, 0] # Green
    overlay = cv2.addWeighted(img_cv, 0.75, color_mask, 0.25, 0)
    
    cx, cy = int(res["center"][0]), int(res["center"][1])
    if res["tail_tip"]:
        tx, ty = int(res["tail_tip"][0]), int(res["tail_tip"][1])
        # Draw arrow from center to tail tip
        cv2.arrowedLine(overlay, (cx, cy), (tx, ty), (0, 0, 255), 3, tipLength=0.3)
        cv2.circle(overlay, (tx, ty), 6, (0, 255, 255), -1)
        
    cv2.imwrite(out_path, overlay)
    print(f"SAM 2 visualization saved to {out_path}!")
    return res

if __name__ == "__main__":
    test_img = "BYTE ML Task Data/test/images/seq_5303b549f62e9d3f/01.png"
    test_box = (444.0, 436.0, 506.0, 498.0)
    out_dir = Path("C:/Users/veer1/.gemini/antigravity-ide/brain/413aacec-e443-4a09-806d-5eac4d796551/scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = str(out_dir / "sam2_balloon_demo.png")
    visualize_sam2_balloon(test_img, test_box, out_img)
