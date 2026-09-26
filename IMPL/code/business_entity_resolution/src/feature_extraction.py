"""
Optimized Feature Extraction Module
Computes high-speed string similarity, token overlap, and distance features for candidate pairs.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
try:
    from .preprocess import clean_name, clean_address, extract_digits
except ImportError:
    from preprocess import clean_name, clean_address, extract_digits


def jaccard_similarity(str1: str, str2: str) -> float:
    """Compute token Jaccard similarity between two strings."""
    set1 = set(str1.split())
    set2 = set(str2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def extract_pair_features(candidate_pairs_df: pd.DataFrame,
                          df_s1: pd.DataFrame,
                          df_target: pd.DataFrame) -> pd.DataFrame:
    """
    Fast feature extractor using direct dictionary lookup and list comprehensions.
    """
    # Create fast dict lookups
    s1_names = dict(zip(df_s1['entity_id'], df_s1['business_name'].fillna('')))
    s1_addrs = dict(zip(df_s1['entity_id'], df_s1['business_address'].fillna('')))

    target_names = dict(zip(df_target['entity_id'], df_target['business_name'].fillna('')))
    target_addrs = dict(zip(df_target['entity_id'], df_target['business_address'].fillna('')))

    s1_ids = candidate_pairs_df['source1_entity_id'].values
    cand_ids = candidate_pairs_df['candidate_entity_id'].values
    block_scores = candidate_pairs_df['blocking_score'].values if 'blocking_score' in candidate_pairs_df else np.ones(len(candidate_pairs_df))
    ranks = candidate_pairs_df['rank'].values if 'rank' in candidate_pairs_df else np.ones(len(candidate_pairs_df))

    # Pre-clean strings in batch for unique entities to avoid re-cleaning
    unique_s1_names = {i: clean_name(str(s1_names.get(i, ''))) for i in set(s1_ids)}
    unique_s1_addrs = {i: clean_address(str(s1_addrs.get(i, ''))) for i in set(s1_ids)}

    unique_target_names = {i: clean_name(str(target_names.get(i, ''))) for i in set(cand_ids)}
    unique_target_addrs = {i: clean_address(str(target_addrs.get(i, ''))) for i in set(cand_ids)}

    n_samples = len(candidate_pairs_df)

    name_ratio = np.zeros(n_samples, dtype=np.float32)
    name_partial = np.zeros(n_samples, dtype=np.float32)
    name_token_sort = np.zeros(n_samples, dtype=np.float32)
    name_token_set = np.zeros(n_samples, dtype=np.float32)
    name_jaccard = np.zeros(n_samples, dtype=np.float32)
    name_exact_clean = np.zeros(n_samples, dtype=np.float32)

    addr_ratio = np.zeros(n_samples, dtype=np.float32)
    addr_partial = np.zeros(n_samples, dtype=np.float32)
    addr_token_sort = np.zeros(n_samples, dtype=np.float32)
    addr_jaccard = np.zeros(n_samples, dtype=np.float32)
    is_s2 = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        s1_i = s1_ids[i]
        c_i = cand_ids[i]

        cn1 = unique_s1_names.get(s1_i, '')
        cn2 = unique_target_names.get(c_i, '')
        ca1 = unique_s1_addrs.get(s1_i, '')
        ca2 = unique_target_addrs.get(c_i, '')

        name_ratio[i] = fuzz.ratio(cn1, cn2) / 100.0
        name_partial[i] = fuzz.partial_ratio(cn1, cn2) / 100.0
        name_token_sort[i] = fuzz.token_sort_ratio(cn1, cn2) / 100.0
        name_token_set[i] = fuzz.token_set_ratio(cn1, cn2) / 100.0
        name_jaccard[i] = jaccard_similarity(cn1, cn2)
        name_exact_clean[i] = 1.0 if (cn1 and cn1 == cn2) else 0.0

        if ca1 and ca2:
            addr_ratio[i] = fuzz.ratio(ca1, ca2) / 100.0
            addr_partial[i] = fuzz.partial_ratio(ca1, ca2) / 100.0
            addr_token_sort[i] = fuzz.token_sort_ratio(ca1, ca2) / 100.0
            addr_jaccard[i] = jaccard_similarity(ca1, ca2)

        is_s2[i] = 1.0 if c_i.startswith('S2') else 0.0

    feat_df = pd.DataFrame({
        'blocking_score': block_scores.astype(np.float32),
        'rank': ranks.astype(np.float32),
        'name_ratio': name_ratio,
        'name_partial': name_partial,
        'name_token_sort': name_token_sort,
        'name_token_set': name_token_set,
        'name_jaccard': name_jaccard,
        'name_exact_clean': name_exact_clean,
        'addr_ratio': addr_ratio,
        'addr_partial': addr_partial,
        'addr_token_sort': addr_token_sort,
        'addr_jaccard': addr_jaccard,
        'is_s2': is_s2
    })

    return feat_df
