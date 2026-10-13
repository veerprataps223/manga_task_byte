"""
Extract Florence-2 OCR with Region for manga pages and cache results.
Uses Microsoft Florence-2-large in float16 on GPU.
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

def init_florence():
    model_id = "microsoft/Florence-2-large"
    print("Loading Florence-2-large processor and model...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        dtype=torch.float16
    ).cuda()
    model.eval()
    return processor, model

def extract_page_ocr(img_path: str, processor, model) -> Dict[str, Any]:
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    
    task_prompt = "<OCR_WITH_REGION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            use_cache=False
        )
        
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(w, h)
    )
    
    raw = parsed_answer.get("<OCR_WITH_REGION>", {})
    quad_boxes = raw.get("quad_boxes", [])
    labels = raw.get("labels", [])
    
    boxes = []
    for quad, label in zip(quad_boxes, labels):
        # quad is [x1, y1, x2, y2, x3, y3, x4, y4]
        clean_text = label.replace("</s>", "").strip()
        if not clean_text:
            continue
        xs = quad[0::2]
        ys = quad[1::2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        boxes.append({
            "text": clean_text,
            "x1": float(min_x),
            "y1": float(min_y),
            "x2": float(max_x),
            "y2": float(max_y),
            "cx": float(sum(xs) / 4.0),
            "cy": float(sum(ys) / 4.0),
            "quad": [float(v) for v in quad]
        })
        
    return {
        "image_path": img_path,
        "width": w,
        "height": h,
        "boxes": boxes
    }

def extract_and_cache_sequences(sequences_json: str, data_dir: str, cache_dir: str, limit: int = None):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    with open(sequences_json, "r", encoding="utf-8") as f:
        seqs = json.load(f)
        
    if limit:
        seqs = seqs[:limit]
        
    processor, model = init_florence()
    print(f"Extracting Florence-2 OCR for {len(seqs)} sequences...")
    
    t_start = time.time()
    for i, seq in enumerate(seqs):
        seq_id = seq["sequence_id"]
        out_file = cache_path / f"{seq_id}.json"
        
        if out_file.exists():
            print(f"[{i+1}/{len(seqs)}] {seq_id} already cached, skipping.")
            continue
            
        t0 = time.time()
        seq_result = {"sequence_id": seq_id, "pages": []}
        for img_rel in seq["images"]:
            img_full = str(Path(data_dir) / img_rel)
            page_data = extract_page_ocr(img_full, processor, model)
            seq_result["pages"].append(page_data)
            
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(seq_result, f, indent=2, ensure_ascii=False)
            
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(seqs)}] {seq_id} done in {elapsed:.1f}s ({len(seq_result['pages'])} pages)")
        
    print(f"All extractions finished in {time.time()-t_start:.1f}s!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=str, default="data/val_sequences.json")
    parser.add_argument("--data_dir", type=str, default="BYTE ML Task Data")
    parser.add_argument("--cache_dir", type=str, default="cache/florence_ocr")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    extract_and_cache_sequences(
        sequences_json=args.sequences,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        limit=args.limit
    )
