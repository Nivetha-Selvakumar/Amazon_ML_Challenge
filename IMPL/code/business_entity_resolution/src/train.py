"""
Ground Truth Informed Training Script
Extracts ground truth matches and hard negative blocking candidates across Source 1, Source 2, and Source 3,
training LightGBM Entity Matching Classifier for optimal Macro F_0.5 performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

try:
    from .preprocess import clean_name, clean_address
    from .blocking import CountryTFIDFBlocker
    from .feature_extraction import extract_pair_features
    from .model import EntityMatchingModel, calculate_macro_f05
except ImportError:
    from preprocess import clean_name, clean_address
    from blocking import CountryTFIDFBlocker
    from feature_extraction import extract_pair_features
    from model import EntityMatchingModel, calculate_macro_f05


def run_training(dataset_dir: str, model_save_path: str, sample_size: int = 15000):
    """
    Train entity resolution model on train dataset using Ground Truth positive sampling
    and TF-IDF hard negative candidate generation.
    """
    print(f"Loading Ground Truth from {dataset_dir} (sample_size={sample_size})...")
    gt_path = os.path.join(dataset_dir, 'train_ground_truth.tsv')
    df_gt = pd.read_csv(gt_path, sep='\t', nrows=sample_size)

    gt_s1_ids = set(df_gt['source1_entity_id'])
    gt_dict = {}
    all_matched_ids = set()
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = [m.strip() for m in str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) and m.strip()]
        gt_dict[s1_id] = set(matches)
        all_matched_ids.update(matches)

    s2_matched_ids = {x for x in all_matched_ids if x.startswith('S2')}
    s3_matched_ids = {x for x in all_matched_ids if x.startswith('S3')}

    print(f"Target GT positive IDs - S1: {len(gt_s1_ids)}, S2: {len(s2_matched_ids)}, S3: {len(s3_matched_ids)}")

    print("Loading Source 1 entities matching Ground Truth...")
    s1_file = os.path.join(dataset_dir, 'train_source1.tsv')
    s1_chunks = []
    for chunk in pd.read_csv(s1_file, sep='\t', chunksize=200000):
        matched = chunk[chunk['entity_id'].isin(gt_s1_ids)]
        s1_chunks.append(matched)
        if sum(len(c) for c in s1_chunks) >= len(gt_s1_ids):
            break
    df_s1 = pd.concat(s1_chunks, ignore_index=True)

    print("Loading Source 2 and Source 3 entities (GT matches + background candidates)...")
    s2_file = os.path.join(dataset_dir, 'train_source2.tsv')
    s2_chunks = [chunk[chunk['entity_id'].isin(s2_matched_ids)] for chunk in pd.read_csv(s2_file, sep='\t', chunksize=500000)]
    s2_chunks.append(pd.read_csv(s2_file, sep='\t', nrows=25000))
    df_s2 = pd.concat(s2_chunks, ignore_index=True).drop_duplicates(subset=['entity_id'])

    s3_file = os.path.join(dataset_dir, 'train_source3.tsv')
    s3_chunks = [chunk[chunk['entity_id'].isin(s3_matched_ids)] for chunk in pd.read_csv(s3_file, sep='\t', chunksize=500000)]
    s3_chunks.append(pd.read_csv(s3_file, sep='\t', nrows=25000))
    df_s3 = pd.concat(s3_chunks, ignore_index=True).drop_duplicates(subset=['entity_id'])

    print(f"Loaded S1: {len(df_s1)}, S2: {len(df_s2)}, S3: {len(df_s3)}")

    print("Running TF-IDF Candidate Blocking...")
    blocker = CountryTFIDFBlocker(top_k=15, min_sim=0.10)
    candidates_df, candidates_dict = blocker.generate_candidates(df_s1, df_s2, df_s3)
    print(f"Generated {len(candidates_df)} candidates via blocking.")

    print("Adding explicit Ground Truth positive pairs to training set...")
    gt_pairs = []
    for s1_id in df_s1['entity_id']:
        for m in gt_dict.get(s1_id, set()):
            gt_pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': m, 'blocking_score': 1.0, 'rank': 1})

    gt_cand_df = pd.DataFrame(gt_pairs)
    combined_cand_df = pd.concat([candidates_df, gt_cand_df], ignore_index=True).drop_duplicates(subset=['source1_entity_id', 'candidate_entity_id'])

    print(f"Total training candidate pairs: {len(combined_cand_df)}")

    print("Extracting pairwise similarity features...")
    df_target = pd.concat([df_s2, df_s3], ignore_index=True)
    X = extract_pair_features(combined_cand_df, df_s1, df_target)

    # Compute binary label y
    y = np.array([
        1.0 if row['candidate_entity_id'] in gt_dict.get(row['source1_entity_id'], set()) else 0.0
        for _, row in combined_cand_df.iterrows()
    ])

    print(f"Training set positives: {int(y.sum())} / {len(y)} ({y.mean():.4f})")

    print("Training LightGBM Entity Matching Classifier...")
    model = EntityMatchingModel()
    model.fit(X, y)

    probas = model.predict_proba(X)
    best_t, best_f05 = model.optimize_threshold(
        combined_cand_df, probas, gt_dict, set(df_s1['entity_id'])
    )

    print(f"Optimal Decision Threshold: {best_t:.3f}")
    print(f"Validation Macro F_0.5 Score: {best_f05:.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Saved model to {model_save_path}")

    return model, best_f05


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_path = os.path.join(base_dir, 'student_resource', 'student_resource', 'dataset', 'train')
    model_path = os.path.join(base_dir, 'code', 'business_entity_resolution', 'src', 'model.pkl')
    run_training(data_path, model_path)
