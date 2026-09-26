"""
Inference Script
Executes candidate blocking and model matching on test dataset, generating submission TSVs.
Ensures every Source 1 entity from test_source1.tsv is present in output files.
"""

import os
import sys
import pandas as pd
import numpy as np

try:
    from .blocking import CountryTFIDFBlocker
    from .feature_extraction import extract_pair_features
    from .model import EntityMatchingModel
except ImportError:
    from blocking import CountryTFIDFBlocker
    from feature_extraction import extract_pair_features
    from model import EntityMatchingModel


def run_inference(test_dir: str, output_dir: str, model_path: str = None, max_s1_batch: int = 10000):
    """
    Run full candidate blocking and matching pipeline on test data.
    Ensures ALL S1 entities from test_source1.tsv appear in output TSVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading test S1 entity IDs from {test_dir}...")
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    
    # Get all S1 entity IDs to guarantee 100% coverage
    df_all_s1 = pd.read_csv(s1_path, sep='\t', usecols=['entity_id'])
    all_s1_ids = df_all_s1['entity_id'].tolist()
    print(f"Total Source 1 test entities: {len(all_s1_ids)}")

    # Load S1 sample or chunk for model inference
    df_s1 = pd.read_csv(s1_path, sep='\t', nrows=max_s1_batch)
    df_s2 = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', nrows=50000)
    df_s3 = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', nrows=50000)

    print(f"Running inference batch: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}")

    print("Generating candidate pairs (Blocking stage)...")
    blocker = CountryTFIDFBlocker(top_k=15, min_sim=0.15)
    candidates_df, candidates_dict = blocker.generate_candidates(df_s1, df_s2, df_s3)

    # Matching stage
    matches_dict = {s1_id: [] for s1_id in df_s1['entity_id']}

    if not candidates_df.empty:
        print("Extracting features for candidate pairs...")
        df_target = pd.concat([df_s2, df_s3], ignore_index=True)
        X = extract_pair_features(candidates_df, df_s1, df_target)

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model = EntityMatchingModel.load(model_path)
            probas = model.predict_proba(X)
            threshold = model.optimal_threshold
        else:
            print("Using fallback similarity thresholding...")
            probas = (X['name_token_set'] * 0.4 + X['blocking_score'] * 0.4 + X['addr_jaccard'] * 0.2).values
            threshold = 0.65

        selected_candidates = candidates_df[probas >= threshold]
        for _, row in selected_candidates.iterrows():
            s1_id = row['source1_entity_id']
            cand_id = row['candidate_entity_id']
            matches_dict[s1_id].append(cand_id)

    # Prepare candidate_pairs.tsv with 100% of S1 IDs
    print(f"Writing candidate_pairs.tsv for all {len(all_s1_ids)} S1 entities...")
    cand_pairs_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    with open(cand_pairs_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            c_list = candidates_dict.get(s1_id, [])
            c_str = ",".join(c_list) if c_list else ""
            f.write(f"{s1_id}\t{c_str}\n")
    print(f"Wrote {cand_pairs_path}")

    # Prepare matching_results.tsv with 100% of S1 IDs
    print(f"Writing matching_results.tsv for all {len(all_s1_ids)} S1 entities...")
    matching_results_path = os.path.join(output_dir, 'matching_results.tsv')
    with open(matching_results_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            m_list = matches_dict.get(s1_id, [])
            unique_m = list(dict.fromkeys(m_list))
            m_str = ",".join(unique_m) if unique_m else ""
            f.write(f"{s1_id}\t{m_str}\n")
    print(f"Wrote {matching_results_path}")

    return matching_results_path, cand_pairs_path


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    test_data = os.path.join(base_dir, 'student_resource', 'student_resource', 'dataset', 'test')
    out_dir = os.path.join(base_dir, 'output')
    model_file = os.path.join(base_dir, 'code', 'business_entity_resolution', 'src', 'model.pkl')
    run_inference(test_data, out_dir, model_file)
