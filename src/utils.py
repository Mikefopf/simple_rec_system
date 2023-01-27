"""Some necessary functions to educate models"""
from typing import List, Dict

from scipy import sparse as sp
from tqdm import tqdm
import numpy as np

from src.metrics import normalized_average_precision

def make_coo_row(items: List[int], n_items: int) -> np.array:
    """Create row for sparse matrix"""
    values = [1.0 for _ in items]
    return sp.coo_matrix(
        (np.array(values).astype(np.float32),
        ([0] * len(value), value)), shape=(1, n_items + 1),
    )

def create_sparse_matrix(
    n_users: int, n_items: int, aggregated_train: Dict[int, List[int]]
) -> np.array:
    """Create sparse matrix"""
    rows = []
    for i in range(1, n_users + 1):
        rows.append(make_coo_row(aggregated_train.get(i, []), n_items))
    return sp.vstack(rows).tocsr()


def aggregate_users(rows: list) -> Dict[int, List[int]]:
    """Aggregate data by users"""
    aggregated_data = {}
    for row in rows:
        if row[0] in aggregated_data:
            aggregated_data[row[0]].append(row[1])
        else:
            aggregated_data[row[0]] = [row[1]]
    return aggregated_data

def get_score_model(
    aggregated_items_test: Dict[int, List[int]], aggregated_items_train: Dict[int, List[int]], 
    model, n_items: int
) -> float:
    """Get score of ml model"""
    scores = []
    for key, value in tqdm(aggregated_items_test.items()):
        row_sparse = make_coo_row(aggregated_items_train.get(key, []), n_items).tocsr()
        recommended_items = model.recommend(
            int(key - 1), row_sparse, N=30, filter_already_liked_items=True, recalculate_user=False
        )
        scores.append(normalized_average_precision(value, recommended_items[0], k=30))
    return np.mean(scores)

def get_score_top(aggregated_items_test: Dict[int, List[int]], top_items: List[int]) -> float:
    """Get score of simple model with top items"""
    scores = []
    for _, value in tqdm(aggregated_items_test.items()):
        scores.append(normalized_average_precision(value, top_items, k=30))
    return np.mean(scores)
