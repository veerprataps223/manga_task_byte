"""
Utility script to display the head of the generated predictions.jsonl file.
Usage:
    python show_head.py
    python show_head.py --file predictions.jsonl --n 2
"""
import json
import argparse
from pathlib import Path

def show_head(filepath: str = "predictions.jsonl", num_sequences: int = 1):
    path = Path(filepath)
    if not path.exists():
        print(f"Error: {filepath} not found. Please run 'python src/predict_test.py' first.")
        return

    print("=" * 70)
    print(f"DISPLAYING HEAD OF PREDICTIONS: {filepath}")
    print("=" * 70)

    with open(path, "r", encoding="utf-8") as f:
        for seq_idx, line in enumerate(f):
            if seq_idx >= num_sequences:
                break
            data = json.loads(line)
            sid = data["sequence_id"]
            pages = data["pages"]
            print(f"\n[Sequence {seq_idx + 1}] ID: {sid} (Total Pages: {len(pages)})")
            print("-" * 70)
            for page_idx, page in enumerate(pages, 1):
                print(f"  Page {page_idx} ({len(page)} dialogue lines):")
                if not page:
                    print("    (Empty page / no dialogue)")
                for item_idx, item in enumerate(page, 1):
                    speaker = item["speaker"]
                    text = item["text"]
                    print(f"    {item_idx:>2}. [{speaker:<10}] {text}")
            print("-" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display head of predictions.jsonl")
    parser.add_argument("--file", type=str, default="predictions.jsonl", help="Prediction JSONL path")
    parser.add_argument("--n", type=int, default=1, help="Number of sequences to display")
    args = parser.parse_args()
    show_head(args.file, args.n)
