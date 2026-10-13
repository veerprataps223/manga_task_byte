Manga Dialogue Extraction and Speaker Attribution

NOTE: this readme is not ai generated. AI is used for assistance like for a better grammar and better explanation of the pipeline, flowchart generated is by AI.
I request you to also have a look towards the main "README.md" which covers everything in detail. but is ai generated with help and corrections from me.

Approach:

We process 3 consecutive manga pages to extract story dialogue in visual reading order and attribute speakers consistently across all 3 pages.

The pipeline runs in 5 stages:
1) OCR: Florence-2 Large extracts text and bounding coordinates, replacing EasyOCR which failed on stylized comic lettering.
2) Spatial NMS: There can be multiple duplications in the final output. This makes sure to remove those duplications and keep the output unique.
3) Noise and Story Filtering: A trained Gradient Boosting classifier removes sound effects, Japanese margin watermarks etc.
4) Reading Order: A trained pairwise ranking model orders speech bubbles because mangas are mainly read right to left, top to bottom, this makes sure everything is in consecutive order, not breaking the flow of speech
5) Speaker Attribution: This separates story text from narration, and spatial clustering groups dialogue into character turns across the 3-page sequence.

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

- The primary pipeline is 15x faster and runs on CPU with zero GPU memory needed during inference. It does run GPU during initial florence stage, to map out bounding boxes.


Time and Latency Benchmark (15 test sequences / 45 pages):

1) Post-OCR inference time (predict_test.py):
Direction A (Primary CPU pipeline): 3.69 s total (82 ms per page, 246 ms per sequence, 0 MB GPU VRAM)
5-Model pipeline (SAM 2 + CLIP GPU): 55.92 s total (1240 ms per page, 3728 ms per sequence, ~2.8 GB GPU VRAM)
Speedup: Direction A is 15.2x faster.

2) End-to-end time from raw images:
Direction A (Florence-2 OCR + CPU pipeline): ~93.7 s total (~2.08 s per page)
5-Model pipeline (Florence-2 OCR + SAM 2 + CLIP): ~145.9 s total (~3.24 s per page)

as you can see from above data , the total time taken by florence 2 ocr + cpu is roughly 36% lesser than 5 model pipeline.

Trained Weights and Open-Source Models:

1) Story Filter Classifier: models/story_filter.pkl (Trained GBDT model, 97.7% CV F1, included in repository)
2) Reading Order Sorter: models/reading_order_sorter.pkl (Trained Pairwise Ranking model, 95.8% CV Accuracy, included in repository)
3) Microsoft Florence-2 Large: https://huggingface.co/microsoft/Florence-2-large (Open weights, 770M params)
4) Meta SAM 2 Hiera Tiny: https://huggingface.co/facebook/sam2-hiera-tiny (Open weights, 38.9M params)
5) OpenAI CLIP ViT-B/32: https://huggingface.co/openai/clip-vit-base-patch32 (Open weights, 87.8M params)


What Worked vs What Failed:

What worked:

1) Foundation model OCR: Florence-2 jumped token coverage from 57% to 83%, fixing broken all-caps comic fonts.(No prior fine-tuning done on florence 2)
2) Spatial NMS: Pruning high-overlap text quads stopped repeated dialogue lines.(Used for removing duplications in final output, i tested this with sequence 1 and 3, 6 pages, every error was avoided)
3) Pairwise ranking: Making sure order of the narratives is being followed
4) SAM 2 tail vectors: Tracing balloon tails to characters raised raw speaker accuracy from 54% to 61%. This makes sure that narrator speaking is accurately chosen(we didnt use this in our final submission)
5) Pronoun rules: Disqualifying bubbles with I, me, you from narration stopped false narration tags.

What failed:

1) Simple aspect-ratio narration rules: Wide boxes at page edges falsely caught Japanese publisher logos and chapter titles.
2) Distance-threshold clustering on raw crops: Inventing extra character clusters crushed Hungarian Balanced F1.


What I Would Try Next

1) Fine-tune Qwen2.5-VL end-to-end to predict OCR, reading order, and speaker tags directly in one pass.
2) Use an anime face detector with ray-casting along the SAM 2 tail vector to snap directly to character face boxes.
3) Add a lightweight dialogue language model prior to enforce realistic turn-taking alternation between speakers.

why we didnt choose 5 model pipeline:

Our priority with this project was to get as much accuracy as we get with efficiency as well.
We prioritised saving time and gpu inference over ~7% extra accurracy.
because that would add roughly 37% extra time, and most of the added overhead is gpu.
On large scale that can create a difference of more than double in $.
So efficiency(lightweightness) > accuracy(~7%)

We did include benchmarks in README_DETAILED.md

How to Run

1) Install dependencies:
pip install -r requirements.txt

2) Train the learned models:
python src/train_models.py

3) Generate test predictions:
python src/predict_test.py --output predictions.jsonl

4) View the head of the output:
python show_head.py

NOTE: 
All the tests were run on RTX 4060(8GB VRAM), i7-13620h, 16gb ram.