# Business Entity Resolution Pipeline

This package contains an end-to-end Machine Learning pipeline for multi-source Business Entity Resolution.

## Architecture

The pipeline consists of a 2-stage entity resolution workflow:

1. **Stage 1: Multi-Index Candidate Generation (Blocking)**
   - Partitioned by country (supporting open-set string country labels including US, India, France, etc.).
   - Sparse TF-IDF character (3-4 gram) and token vectorization.
   - High-speed cosine similarity matrix multiplication pruned by similarity threshold.

2. **Stage 2: Pairwise Feature Extraction & Gradient Boosted Classification**
   - Pairwise string similarity metrics: RapidFuzz ratio, partial ratio, token sort ratio, token set ratio.
   - Address token Jaccard similarity and numeric PIN/street number digit matching.
   - LightGBM Gradient Boosted Decision Trees for binary match classification.
   - $F_{0.5}$ metric grid-search threshold optimization to maximize Precision-heavy evaluation.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Model Training (Validation & Threshold Tuning):
   ```bash
   python -m src.train
   ```

3. Run Test Inference & Generate Submissions:
   ```bash
   python -m src.predict
   ```

4. Validate Submission Output Format:
   ```bash
   python ../../student_resource/student_resource/utils/validate_submission.py \
       --matching ../../output/matching_results.tsv \
       --candidate ../../output/candidate_pairs.tsv \
       --test-dir ../../student_resource/student_resource/dataset/test
   ```
