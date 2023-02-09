"""
Some necessary functions to educate models
"""
from typing import List, Dict, Iterable, Any
from itertools import product

from scipy import sparse as sp
import numpy as np

from metrics import normalized_average_precision

def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    """
    Get grid parameters using functools.
    
    Arguments:
        parameters:
            Dict: parameter_name: grid for the parameter.
    
    Return:
        Generator of dicts parameter_name: parameter.
    """
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))

def make_coo_row(items: List[int], n_items: int) -> np.array:
    """
    Create row for sparse matrix
    
    item_(j) = 1 if user bought item j,
    else item_(j) = 0
    
    Arguments:
        items:
            List of items that bought user.
        n_items:
            Number of items in history.
    
    Return:
        Coo row
    """
    values = [1.0 for _ in items]
    return sp.coo_matrix(
        (np.array(values).astype(np.float32),
        ([0] * len(items), items)), shape=(1, n_items + 1),
    )

def create_sparse_matrix(
    n_users: int, n_items: int, aggregated_train: Dict[int, List[int]]
) -> np.array:
    """Create sparse matrix by rows
    
    This matrix based on train data of purchases.
    Number of rows = users number.
    Number of columns = items number.
    
    Arguments:
        n_users:
            Number of users in history.
        n_items:
            Number of items in history.
        aggregated_train:
            All transactions in train dataset aggregated by users.
            Dict: {user: items of users}
    
    Return:
        Sparse matrix.
    """
    rows = [make_coo_row(aggregated_train.get(user, []), n_items) 
            for user in range(1, n_users + 1)]
    return sp.vstack(rows).tocsr()


def aggregate_users(rows: list) -> Dict[int, List[int]]:
    """
    Aggregate data by users
    
    For every user get list of his purchases and concatenate
    it for all users.
    
    Return:
        Dict: {user: items of users}
    """
    aggregated_data = {}
    for row in rows:
        user, item = row[0], row[1]
        current_items = aggregated_data.get(user, [])
        current_items.append(item)
        aggregated_data[user] = current_items
    return aggregated_data

def get_score_model(
    aggregated_items_test: Dict[int, List[int]], aggregated_items_train: Dict[int, List[int]], 
    model, n_items: int
) -> float:
    """
    Get score of ml model.
    
    Return:
        Mean score for users.
    """
    scores = []
    for key, value in aggregated_items_test.items():
        row_sparse = make_coo_row(aggregated_items_train.get(key, []), n_items).tocsr()
        recommended_items = model.recommend(
            int(key - 1), row_sparse, N=30, filter_already_liked_items=True, recalculate_user=False
        )
        scores.append(normalized_average_precision(value, recommended_items[0], k=30))
    return np.mean(scores)

def get_score_top(aggregated_items_test: Dict[int, List[int]], top_items: List[int]) -> float:
    """
    Get score of simple model with top items
    
    Return:
        Mean score for users.
    """
    scores = [normalized_average_precision(value, top_items, k=30)
              for value in aggregated_items_test.values()]
    return np.mean(scores)
    
def grid_search_recommend(
    model, grid_hyperparams: dict, train: dict, val: dict) -> dict:
    """
    Grid search.
    
    Arguments:
        model:
            Model to fit.
        grid_hyperparams:
            Full grid:
                Dict: {parameter_name: grid}
        train:
            Train aggregated dataset
        val:
            Val aggregated dataset
    
    Return:
        Best parameters
    """
    best_score = -1.0
    for settings in grid_parameters(grid_hyperparams):
        X_sparse = create_sparse_matrix(30910, 18494, train)
        model_params = model(**settings)
        model_params.fit(X_sparse)
        score = get_score_model(val, train, model_params, 18494)
        if score > best_score:
            best_score = score
            best_params = settings
    return best_params
        