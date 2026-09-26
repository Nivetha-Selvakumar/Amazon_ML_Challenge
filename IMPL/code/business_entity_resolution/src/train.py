"""
Streamlined Training Script
Fast loading for entity resolution training.
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


def run_training(dataset_dir: str, model_save_path: str, sample_size: int = 2000):
    """
    Train entity resolution model on train dataset sample.
    """
    print(f"Loading training dataset from {dataset_dir} (sample_size={sample_size})...")
    df_s1 = pd.read_csv(os.path.join(dataset_dir, 'train_source1.tsv'), sep='\t', nrows=sample_size)
    df_s2 = pd.read_csv(os.path.join(dataset_dir, 'train_source2.tsv'), sep='\t', nrows=50000)
    df_s3 = pd.read_csv(os.path.join(dataset_dir, 'train_source3.tsv'), sep='\t', nrows=50000)
    df_gt = pd.read_csv(os.path.join(dataset_dir, 'train_ground_truth.tsv'), sep='\t', nrows=sample_size)

    s1_set = set(df_s1['entity_id'])
    df_gt = df_gt[df_gt['source1_entity_id'].isin(s1_set)].reset_index(drop=True)

    gt_dict = {}
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip() else []
        gt_dict[s1_id] = set(m for m in matches if m)

    print(f"Running candidate blocking for {len(df_s1)} S1 entities...")
    blocker = CountryTFIDFBlocker(top_k=15, min_sim=0.15)
    candidates_df, candidates_dict = blocker.generate_candidates(df_s1, df_s2, df_s3)
    print(f"Generated {len(candidates_df)} candidate pairs.")

    print("Extracting pairwise similarity features...")
    df_target = pd.concat([df_s2, df_s3], ignore_index=True)
    X = extract_pair_features(candidates_df, df_s1, df_target)

    # Compute binary label y
    y = []
    for _, row in candidates_df.iterrows():
        s1_id = row['source1_entity_id']
        cand_id = row['candidate_entity_id']
        is_match = 1.0 if cand_id in gt_dict.get(s1_id, set()) else 0.0
        y.append(is_match)
    y = np.array(y)

    print(f"Training set positives: {int(y.sum())} / {len(y)} ({y.mean():.4f})")

    print("Training LightGBM Entity Matching Classifier...")
    model = EntityMatchingModel()
    model.fit(X, y)

    probas = model.predict_proba(X)
    best_t, best_f05 = model.optimize_threshold(
        candidates_df, probas, gt_dict, set(df_s1['entity_id'])
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
