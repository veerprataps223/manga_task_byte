import json
import random
from pathlib import Path

def create_splits(dev_labels_path: str, sequences_path: str, output_dir: str, seed: int = 42, val_size: int = 15):
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(dev_labels_path, "r", encoding="utf-8") as f:
        dev_labels = [json.loads(line) for line in f if line.strip()]
        
    with open(sequences_path, "r", encoding="utf-8") as f:
        all_sequences = json.load(f)
        
    dev_seq_map = {s["sequence_id"]: s for s in all_sequences if s["split"] == "development"}
    test_seqs = [s for s in all_sequences if s["split"] == "test"]
    
    # Randomly shuffle and split dev sequences
    dev_seq_ids = [row["sequence_id"] for row in dev_labels]
    random.shuffle(dev_seq_ids)
    
    val_ids = set(dev_seq_ids[:val_size])
    train_ids = set(dev_seq_ids[val_size:])
    
    train_labels = [row for row in dev_labels if row["sequence_id"] in train_ids]
    val_labels = [row for row in dev_labels if row["sequence_id"] in val_ids]
    
    train_seqs = [dev_seq_map[sid] for sid in dev_seq_ids if sid in train_ids]
    val_seqs = [dev_seq_map[sid] for sid in dev_seq_ids if sid in val_ids]
    
    # Write train/val labels
    with open(output_path / "train_labels.jsonl", "w", encoding="utf-8") as f:
        for row in train_labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    with open(output_path / "val_labels.jsonl", "w", encoding="utf-8") as f:
        for row in val_labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    # Write sequences
    with open(output_path / "train_sequences.json", "w", encoding="utf-8") as f:
        json.dump(train_seqs, f, indent=2, ensure_ascii=False)
        
    with open(output_path / "val_sequences.json", "w", encoding="utf-8") as f:
        json.dump(val_seqs, f, indent=2, ensure_ascii=False)
        
    with open(output_path / "test_sequences.json", "w", encoding="utf-8") as f:
        json.dump(test_seqs, f, indent=2, ensure_ascii=False)
        
    print(f"Data split successfully created!")
    print(f"Train sequences: {len(train_seqs)}, Val sequences: {len(val_seqs)}, Test sequences: {len(test_seqs)}")
    print(f"Val IDs: {sorted(list(val_ids))}")

if __name__ == "__main__":
    create_splits(
        dev_labels_path="BYTE ML Task Data/development/labels.jsonl",
        sequences_path="BYTE ML Task Data/sequences.json",
        output_dir="data",
        seed=42,
        val_size=15
    )
