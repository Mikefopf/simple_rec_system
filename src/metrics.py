"""Metrics to calculate"""

def average_precision(actual: list, recommended: list, k=30) -> float:
    """Average precision for one user"""
    ap_sum = 0
    hits = 0
    for i in range(k):
        product_id = recommended[i] if i < len(recommended) else None
        if product_id is not None and product_id in actual:
            hits += 1
            ap_sum += hits / (i + 1)
    return ap_sum / k



def normalized_average_precision(actual: list, recommended: list, k=30) -> float:
    """Average precision for all users"""
    actual = set(actual)
    if len(actual) == 0:
        return 0.0

    aver_prec = average_precision(actual, recommended, k=k)
    aver_prec_ideal = average_precision(actual, list(actual)[:k], k=k)
    return aver_prec / aver_prec_ideal
