"""
Inference script to generate final test predictions for the 15 test sequences.
Uses the trained Florence-2 + Learned Manga Processor pipeline.
Outputs predictions in the required JSONL format and validates against score.py rules.
"""
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(".").resolve()))
sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str(Path("BYTE ML Task Data").resolve()))

from manga_processor import merge_word_boxes, sort_manga_order, attribute_speakers_for_sequence
from score import load_jsonl

def generate_test_predictions(test_sequences_path: str, cache_dir: str, output_jsonl_path: str):
    with open(test_sequences_path, "r", encoding="utf-8") as f:
        test_seqs = json.load(f)
        
    cache_path = Path(cache_dir)
    results = []
    
    print(f"Generating test predictions for {len(test_seqs)} sequences from cache: {cache_dir}...")
    for seq in test_seqs:
        sid = seq["sequence_id"]
        cached_file = cache_path / f"{sid}.json"
        
        if not cached_file.exists():
            raise FileNotFoundError(f"Missing cached OCR for test sequence {sid} at {cached_file}")
            
        with open(cached_file, "r", encoding="utf-8") as f:
            seq_data = json.load(f)
            
        pages_bubbles = []
        page_w = 900.0
        page_h = 1300.0
        
        for page in seq_data["pages"]:
            w, h = page["width"], page["height"]
            page_w, page_h = w, h
            # 1. Bubble Clustering & Learned Filtering
            bubbles = merge_word_boxes(page["boxes"], w, h)
            # 2. Learned Pairwise Manga Reading Order Sorting
            ordered_bubbles = sort_manga_order(bubbles, w, h)
            pages_bubbles.append(ordered_bubbles)
            
        # 3. Learned Speaker Attribution & Cross-Page Identity Consistency
        final_pages = attribute_speakers_for_sequence(pages_bubbles, page_w, page_h)
        
        results.append({
            "sequence_id": sid,
            "pages": final_pages
        })
        
    # Write to JSONL
    out_path = Path(output_jsonl_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"Predictions successfully written to {output_jsonl_path} ({len(results)} sequences)")
    
    # Validate format using score.py's load_jsonl
    print("Validating generated predictions against score.py loader...")
    validated = load_jsonl(output_jsonl_path, pages=3)
    print(f"Validation PASSED! All {len(validated)} sequences strictly comply with evaluation format.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_sequences", type=str, default="data/test_sequences.json")
    parser.add_argument("--cache_dir", type=str, default="cache/florence_ocr_test")
    parser.add_argument("--output", type=str, default="predictions.jsonl")
    args = parser.parse_args()
    
    generate_test_predictions(args.test_sequences, args.cache_dir, args.output)
