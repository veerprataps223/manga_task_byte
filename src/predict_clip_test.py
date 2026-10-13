import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path("src").resolve()))
sys.path.append(str(Path("BYTE ML Task Data").resolve()))

from manga_processor import merge_word_boxes, sort_manga_order
from learned_speaker import LearnedSpeakerTracker
from clip_speaker import attribute_sequence_with_clip
from score import load_jsonl

speaker_tracker = LearnedSpeakerTracker()

with open("data/test_sequences.json", encoding="utf-8") as f:
    test_seqs = json.load(f)

cache_dir = Path("cache/florence_ocr_test")
results = []

print(f"Generating CLIP-enhanced test predictions for {len(test_seqs)} sequences...")
t0 = time.time()

for idx, seq in enumerate(test_seqs):
    sid = seq["sequence_id"]
    cached_file = cache_dir / f"{sid}.json"
    with open(cached_file, encoding="utf-8") as f:
        seq_data = json.load(f)
        
    pages_bubbles = []
    page_w, page_h = 900.0, 1300.0
    for page in seq_data["pages"]:
        w, h = page["width"], page["height"]
        page_w, page_h = w, h
        bubbles = merge_word_boxes(page["boxes"], w, h)
        ordered_bubbles = sort_manga_order(bubbles, w, h)
        for b in ordered_bubbles:
            if speaker_tracker.is_narration(b, w, h):
                b["assigned_speaker"] = "NARRATION"
        pages_bubbles.append(ordered_bubbles)
        
    img_paths = [str(Path("BYTE ML Task Data") / rel_p) for rel_p in seq["images"]]
    final_pages = attribute_sequence_with_clip(img_paths, pages_bubbles, page_w, page_h)
    results.append({"sequence_id": sid, "pages": final_pages})

output_path = "predictions_clip.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Predictions successfully written to {output_path} in {time.time()-t0:.2f}s!")
validated = load_jsonl(output_path, pages=3)
print(f"Validation PASSED! All {len(validated)} sequences strictly comply with evaluation format.")
