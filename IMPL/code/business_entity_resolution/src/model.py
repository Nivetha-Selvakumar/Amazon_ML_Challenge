"""
Model Training and F_0.5 Optimization Module
Trains LightGBM Classifier on candidate pair features and optimizes decision threshold
for Macro F_0.5 evaluation metric.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib


def calculate_macro_f05(ground_truth_dict: dict, predictions_dict: dict, all_s1_ids: set) -> float:
    """
    Compute macro-averaged F_0.5 score across all Source 1 entities.

    ground_truth_dict: {s1_id: set([s2/s3 ids])}
    predictions_dict: {s1_id: set([predicted s2/s3 ids])}
    all_s1_ids: set of all S1 entity IDs in evaluation set
    """
    scores = []
    for s1_id in all_s1_ids:
        y_true = ground_truth_dict.get(s1_id, set())
        y_pred = predictions_dict.get(s1_id, set())

        len_true = len(y_true)
        len_pred = len(y_pred)

        if len_true == 0:
            scores.append(1.0 if len_pred == 0 else 0.0)
            continue

        if len_pred == 0:
            scores.append(0.0)
            continue

        overlap = len(y_true.intersection(y_pred))
        precision = overlap / len_pred
        recall = overlap / len_true

        if precision + recall == 0:
            scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            scores.append(f05)

    return float(np.mean(scores))


class EntityMatchingModel:
    """
    Gradient Boosted Matching Model using LightGBM.
    """
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.optimal_threshold = 0.55

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train LightGBM binary classifier."""
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return match probability for each candidate pair."""
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(self, candidate_pairs_df: pd.DataFrame,
                           probas: np.ndarray,
                           ground_truth_dict: dict,
                           all_s1_ids: set):
        """
        Grid search for decision threshold that maximizes Macro F_0.5.
        """
        best_f05 = -1.0
        best_t = 0.5

        thresholds = np.linspace(0.20, 0.85, 27)
        for t in thresholds:
            preds_df = candidate_pairs_df[probas >= t]
            preds_dict = {s1_id: set() for s1_id in all_s1_ids}
            for _, row in preds_df.iterrows():
                s1_id = row['source1_entity_id']
                if s1_id in preds_dict:
                    preds_dict[s1_id].add(row['candidate_entity_id'])
                else:
                    preds_dict[s1_id] = {row['candidate_entity_id']}

            f05 = calculate_macro_f05(ground_truth_dict, preds_dict, all_s1_ids)
            if f05 > best_f05:
                best_f05 = f05
                best_t = float(t)

        self.optimal_threshold = best_t
        return best_t, best_f05

    def save(self, filepath: str):
        joblib.dump({'model': self.model, 'threshold': self.optimal_threshold}, filepath)

    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data['model']
        instance.optimal_threshold = data['threshold']
        return instance
