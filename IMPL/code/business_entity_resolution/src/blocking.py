"""
Blocking / Candidate Generation Module
Generates high-recall candidate matches for Source 1 entities using TF-IDF N-gram similarity
and Token Indexing partitioned by country.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import gc

try:
    from .preprocess import clean_name, clean_address
except ImportError:
    from preprocess import clean_name, clean_address


class CountryTFIDFBlocker:
    """
    Multi-stage Blocker that generates candidate pairs per country partition
    using char n-gram and word TF-IDF sparse similarity.
    """
    def __init__(self, top_k=20, min_sim=0.15, max_features=100000):
        self.top_k = top_k
        self.min_sim = min_sim
        self.max_features = max_features

    def generate_candidates(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame):
        """
        Generate candidate pairs for S1 entities matching S2 and S3.
        Returns:
            candidates_df: DataFrame with ['source1_entity_id', 'candidate_entity_id', 'blocking_score', 'rank']
            candidates_dict: dict mapping s1_id -> list of candidate_ids
        """
        # Ensure country field exists and clean text
        df_s1 = df_s1.copy()
        df_s2 = df_s2.copy()
        df_s3 = df_s3.copy()

        df_s1['clean_text'] = (df_s1['business_name'].fillna('').apply(clean_name) + " " +
                               df_s1['business_address'].fillna('').apply(clean_address)).str.strip()
        df_s2['clean_text'] = (df_s2['business_name'].fillna('').apply(clean_name) + " " +
                               df_s2['business_address'].fillna('').apply(clean_address)).str.strip()
        df_s3['clean_text'] = (df_s3['business_name'].fillna('').apply(clean_name) + " " +
                               df_s3['business_address'].fillna('').apply(clean_address)).str.strip()

        # Combine S2 and S3 for matching
        df_target = pd.concat([df_s2, df_s3], ignore_index=True)

        countries = set(df_s1['country'].dropna().unique()).union(set(df_target['country'].dropna().unique()))
        
        all_candidate_rows = []
        candidates_dict = {s1_id: [] for s1_id in df_s1['entity_id']}

        for country in countries:
            sub_s1 = df_s1[df_s1['country'] == country]
            sub_target = df_target[df_target['country'] == country]

            if sub_s1.empty or sub_target.empty:
                continue

            # Fit TF-IDF Vectorizer on combined text for the country
            vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 4),
                min_df=2,
                max_features=self.max_features,
                dtype=np.float32
            )

            corpus = pd.concat([sub_s1['clean_text'], sub_target['clean_text']], ignore_index=True)
            vectorizer.fit(corpus)

            mat_s1 = vectorizer.transform(sub_s1['clean_text'])
            mat_target = vectorizer.transform(sub_target['clean_text'])

            s1_ids = sub_s1['entity_id'].values
            target_ids = sub_target['entity_id'].values

            # Process S1 in chunks to maintain low memory overhead
            chunk_size = 5000
            num_s1 = mat_s1.shape[0]

            for start_idx in range(0, num_s1, chunk_size):
                end_idx = min(start_idx + chunk_size, num_s1)
                chunk_s1 = mat_s1[start_idx:end_idx]

                # Sparse matrix multiplication for cosine similarity
                sim_matrix = chunk_s1.dot(mat_target.T)  # shape: (chunk_size, num_target)

                for i in range(chunk_s1.shape[0]):
                    global_s1_idx = start_idx + i
                    cur_s1_id = s1_ids[global_s1_idx]

                    row = sim_matrix.getrow(i)
                    if row.nnz == 0:
                        continue

                    # Get indices and values
                    data = row.data
                    indices = row.indices

                    # Filter by min_sim and pick top_k
                    mask = data >= self.min_sim
                    valid_data = data[mask]
                    valid_indices = indices[mask]

                    if len(valid_data) == 0:
                        # Fallback: pick top 3 highest even if < min_sim
                        top_n = min(3, len(data))
                        if top_n > 0:
                            top_idx = np.argpartition(data, -top_n)[-top_n:]
                            top_idx = top_idx[np.argsort(-data[top_idx])]
                            valid_data = data[top_idx]
                            valid_indices = indices[top_idx]

                    if len(valid_data) > 0:
                        # Sort descending
                        sort_order = np.argsort(-valid_data)[:self.top_k]
                        top_indices = valid_indices[sort_order]
                        top_scores = valid_data[sort_order]

                        c_ids = target_ids[top_indices].tolist()
                        candidates_dict[cur_s1_id] = c_ids

                        for rank, (cand_id, score) in enumerate(zip(c_ids, top_scores), start=1):
                            all_candidate_rows.append({
                                'source1_entity_id': cur_s1_id,
                                'candidate_entity_id': cand_id,
                                'blocking_score': float(score),
                                'rank': rank
                            })

            gc.collect()

        candidates_df = pd.DataFrame(all_candidate_rows)
        return candidates_df, candidates_dict
