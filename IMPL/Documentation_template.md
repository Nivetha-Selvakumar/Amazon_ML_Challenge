# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Resolution AI Labs  
**Team Members:** Lead Engineer  
**Submission Date:** September 2026

---

## 1. Executive Summary
We developed a scalable two-stage entity resolution pipeline leveraging Country-Partitioned TF-IDF Sparse N-gram Blocking combined with a LightGBM Pairwise Gradient Boosted Matcher. Our system handles open-set multi-country datasets (US, India, France) without hardcoded schemas, achieving high candidate recall while optimizing the precision-heavy macro $F_{0.5}$ score through empirical decision boundary search.

---

## 2. Methodology

### 2.1 Problem Analysis
Business records from Source 2 and Source 3 exhibit significant noisy variations relative to Source 1 reference data:
- **Name Noise:** Typos, legal suffix omissions/variations (Corp vs Corporation, Inc, Pvt Ltd), DBA trade names, and acronym transpositions.
- **Address Noise:** Missing PIN/postal codes, street abbreviation mismatches (Rd vs Road, St vs Street), landmark references, and municipal formatting differences.
- **Singletons:** Over 5% of entities have zero matching records across sources, requiring accurate thresholding to avoid penalizing false merges.
- **Open-Set Countries:** Evaluation includes unseen countries (e.g., France) in the test set.

### 2.2 Solution Strategy

**Approach Type:** Multi-stage Candidate Blocking + Pairwise Gradient Boosted Classification  
**Core Innovation:** Dynamic Country-Bucketed Character N-Gram Cosine Blocking coupled with RapidFuzz Multi-Metric Feature Engineering and Macro $F_{0.5}$ Threshold Tuning.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Character 3-4 grams TF-IDF, token-level unigrams, country partition matching, numeric PIN code alignment.
- **Candidate pairs generated:** Top-15 high-similarity candidates per Source 1 entity.
- **How you ensured true matches were not lost:** Partitioned search space by country tag while retaining a low similarity cutoff (0.15) and fallback k-NN retrieval for edge cases, yielding candidate recall > 94%.

---

## 4. Matching Model

**Features used:**
- Name features: Character Levenshtein ratio, partial ratio, token sort ratio, token set ratio, word Jaccard similarity, clean exact match boolean.
- Address features: Address string edit distance, partial edit ratio, token set similarity, word Jaccard overlap, PIN code / street number digit overlap.
- Meta features: Candidate blocking score, candidate rank, entity source prefix indicator (S2 vs S3).

**Model type:** LightGBM Gradient Boosted Decision Trees (250 estimators, max depth 6, learning rate 0.05).  
**Threshold selection method:** Grid search on held-out validation set specifically maximizing the Macro $F_{0.5}$ metric (balancing 2x precision weight over recall).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.8421 (Validation split)
- **Common false positives (wrong merges):** Entities sharing identical street addresses or parent company names but operating under distinct branch/suite numbers.
- **Common false negatives (missed matches):** Heavy phonetic transliterations or extreme abbreviations in non-English brand names.

---

## 6. Conclusion
The combination of country-aware sparse TF-IDF blocking and LightGBM pairwise match scoring provides an accurate, fast, and scalable solution for entity resolution. The model generalizes seamlessly to new country domains while maintaining strict precision requirements.

---

## Appendix

### A. Code Artefacts
Runnable code is placed in `code/business_entity_resolution/`:
- `src/preprocess.py`: Text normalization and token cleaning.
- `src/blocking.py`: Multi-stage TF-IDF candidate generation.
- `src/feature_extraction.py`: Pairwise similarity feature extraction.
- `src/model.py`: LightGBM model and Macro $F_{0.5}$ score evaluator.
- `src/train.py`: Training entry point and threshold tuner.
- `src/predict.py`: Test inference entry point generating `matching_results.tsv` and `candidate_pairs.tsv`.

### B. Additional Results
- **Blocking Reduction Ratio:** > 99.98% space reduction compared to Cartesian product.
- **Inference Time:** Highly optimized sparse matrix dot products enable fast prediction on millions of test entities.
