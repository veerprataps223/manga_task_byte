Manga Dialogue Extraction and Speaker Attribution

Approach:

We process 3 consecutive manga pages to extract story dialogue in visual reading order and attribute speakers consistently across all 3 pages.

The pipeline runs in 5 stages:
1) OCR: Florence-2 Large extracts text and bounding coordinates, replacing EasyOCR which failed on stylized comic lettering.
2) Spatial NMS: Merges overlapping text candidate boxes (IoU > 0.45) to eliminate duplicate line emissions.
3) Noise and Story Filtering: A trained Gradient Boosting classifier (16 layout and lexical features) removes sound effects, Japanese margin watermarks, and scanlator notes.
4) Reading Order: A trained pairwise ranking model orders speech bubbles based on vertical tiers and right-to-left layout weights (dy = +10.03, dx = -1.57).
5) Speaker Attribution: A lexical narrator classifier separates story text from narration, and spatial clustering groups dialogue into character turns across the 3-page sequence.

![Pipeline Architecture Flowchart](./assets/pipeline_flowchart.png)


Experiments and Results:

Evaluated on the "15" validation sequences using the official score.py benchmark:

Baseline (EasyOCR + heuristic):
Token Coverage: 57.2%
Text Order Score: 56.5%
Speaker Accuracy: 54.3%
Balanced Joint F1: 0.2063

Florence-2 Raw OCR:
Token Coverage: 85.5%
Text Order Score: 69.9%
Speaker Accuracy: 54.3%
Balanced Joint F1: 0.2943

Learned Pipeline (GBDT + Pairwise Sorter + Spatial NMS):
Token Coverage: 83.1%
Text Order Score: 75.0%
Speaker Accuracy: 56.5%
Balanced Joint F1: 0.2950

Final Primary Model (predictions.jsonl):
Token Coverage: 82.8%
Text Order Score: 75.9%
Speaker Accuracy: 55.0%
Balanced Joint F1: 0.3151
Latency per page: 82 ms on CPU

SAM 2 + CLIP Vision Re-ID (predictions_clip.jsonl):
Token Coverage: 82.8%
Text Order Score: 75.9%
Speaker Accuracy: 61.1%
Balanced Joint F1: 0.2538
Latency per page: 1240 ms on GPU

Why the primary submission uses the CPU pipeline:
1) The official metric heavily penalizes unmapped predicted characters (extra_speakers in the denominator). Vision embeddings occasionally split character clusters, dropping Balanced F1 from 0.3151 to 0.2538.
2) The primary pipeline is 15x faster and runs on CPU with zero GPU memory needed during inference.


What Worked vs What Failed:

What worked:

1) Foundation model OCR: Florence-2 jumped token coverage from 57% to 83%, fixing broken all-caps comic fonts.
2) Spatial NMS: Pruning high-overlap text quads stopped repeated dialogue lines.
3) Pairwise ranking: Learning vertical tier and RTL weights beat rigid box heuristics on borderless panels.
4) SAM 2 tail vectors: Tracing balloon tails to characters raised raw speaker accuracy from 54% to 61%.
5) Pronoun rules: Disqualifying bubbles with I, me, you from narration stopped false narration tags.

What failed:

1) Morphological gutter splitting: Thresholding and contour slicing broke on diagonal gutters and art breaking across borders.
2) Simple aspect-ratio narration rules: Wide boxes at page edges falsely caught Japanese publisher logos and chapter titles.
3) Distance-threshold clustering on raw crops: Inventing extra character clusters crushed Hungarian Balanced F1.


What I Would Try Next

1) Fine-tune Qwen2.5-VL end-to-end to predict OCR, reading order, and speaker tags directly in one pass.
2) Use an anime face detector with ray-casting along the SAM 2 tail vector to snap directly to character face boxes.
3) Add a lightweight dialogue language model prior to enforce realistic turn-taking alternation between speakers.


How to Run

1) Install dependencies:
pip install -r requirements.txt

2) Train the learned models:
python src/train_models.py

3) Generate test predictions:
python src/predict_test.py --output predictions.jsonl

4) View the head of the output:
python show_head.py
