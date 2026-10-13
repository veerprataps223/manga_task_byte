"""
Train Machine Learning Models for Manga Task:
1. Story Filter (Gradient Boosting Classifier)
2. Pairwise Manga Reading Order Sorter (Logistic Regression)
Performs 5-fold cross validation, prints performance metrics, and saves model artifacts.
"""
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

from build_training_data import build_training_dataset

def train_and_save_models(cache_dir: str, labels_jsonl: str, output_model_dir: str):
    out_dir = Path(output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Building dataset from cached sequences and labels...")
    X_filter, y_filter, X_pairs, y_pairs = build_training_dataset(cache_dir, labels_jsonl)
    
    if len(X_filter) == 0:
        print("No training samples found!")
        return
        
    print("\n" + "="*50)
    print("TRAINING MODEL 1: Learned Story Filter Classifier")
    print("="*50)
    
    # 5-fold cross-validation on Story Filter
    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y_filter))), shuffle=True, random_state=42)
    clf_filter = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    cv_scores = cross_val_score(clf_filter, X_filter, y_filter, cv=cv, scoring="f1")
    print(f"Filter 5-fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    clf_filter.fit(X_filter, y_filter)
    y_pred_filter = clf_filter.predict(X_filter)
    print("\nFilter Training Set Report:")
    print(classification_report(y_filter, y_pred_filter, target_names=["Noise/SFX", "Story Dialogue"]))
    
    filter_path = out_dir / "story_filter.pkl"
    with open(filter_path, "wb") as f:
        pickle.dump(clf_filter, f)
    print(f"Saved Filter model to {filter_path}")
    
    print("\n" + "="*50)
    print("TRAINING MODEL 2: Learned Pairwise Reading Order Sorter")
    print("="*50)
    
    clf_sorter = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    cv_sorter_scores = cross_val_score(clf_sorter, X_pairs, y_pairs, cv=5, scoring="accuracy")
    print(f"Sorter 5-fold CV Accuracy: {cv_sorter_scores.mean():.4f} (+/- {cv_sorter_scores.std():.4f})")
    
    clf_sorter.fit(X_pairs, y_pairs)
    y_pred_sorter = clf_sorter.predict(X_pairs)
    print("\nSorter Training Set Report:")
    print(classification_report(y_pairs, y_pred_sorter, target_names=["B before A", "A before B"]))
    print("Learned Feature Weights for Manga Reading Order:")
    feature_names = [
        "dy (B_y - A_y)", "dx (B_x - A_x)", "abs(dy)", "abs(dx)", 
        "overlap_y", "overlap_x", "is_same_row", "is_same_col", 
        "right_before_left", "top_before_bottom"
    ]
    for name, weight in zip(feature_names, clf_sorter.coef_[0]):
        print(f"  {name:22s}: {weight:+.4f}")
        
    sorter_path = out_dir / "reading_order_sorter.pkl"
    with open(sorter_path, "wb") as f:
        pickle.dump(clf_sorter, f)
    print(f"Saved Sorter model to {sorter_path}")

if __name__ == "__main__":
    train_and_save_models(
        cache_dir="cache/florence_ocr",
        labels_jsonl="data/val_labels.jsonl",
        output_model_dir="models"
    )
