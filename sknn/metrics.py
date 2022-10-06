import numpy as np


def hit_rate(predicted_items: np.ndarray, real_items: np.ndarray) -> float:
    hits = 0
    for real, pred in zip(real_items, predicted_items):

        real = real[real > 0]
        pred = pred[pred > 0]

        real_found_in_pred = np.isin(pred, real, assume_unique=True)

        if real_found_in_pred.any():
            hits += 1

    hit_rate = hits / len(predicted_items)
    return hit_rate


def mean_reciprocal_rank(predicted_items: np.ndarray, real_items: np.ndarray) -> float:
    idx = 0
    reciprocal_rank = np.zeros(len(predicted_items), dtype=np.float64)
    for real, pred in zip(real_items, predicted_items):

        real = real[real > 0]
        pred = pred[pred > 0]

        real_found_in_pred = np.isin(pred, real, assume_unique=True)

        if real_found_in_pred.any():
            rank = np.argwhere(real_found_in_pred).ravel()[0] + 1
            curr_recip_rank = 1 / rank
        else:
            curr_recip_rank = 0

        reciprocal_rank[idx] += curr_recip_rank

    # ofc this is not efficient by any means, but I plan to reuse this for
    # another purpose later
    mrr = reciprocal_rank.mean()
    return mrr


def precision(predicted_items: np.ndarray, real_items: np.ndarray) -> float:
    idx = 0
    precisions = np.zeros(len(predicted_items), dtype=np.float64)
    for real, pred in zip(real_items, predicted_items):

        real = real[real > 0]
        pred = pred[pred > 0]

        real_found_in_pred = np.isin(pred, real, assume_unique=True)

        if real_found_in_pred.any():
            recommended = real_found_in_pred.sum()
            precision = recommended / len(pred)
        else:
            precision = 0
        precisions[idx] += precision
        idx += 1
    return precisions.mean()


def recall(predicted_items: np.ndarray, real_items: np.ndarray) -> float:
    idx = 0
    recalls = np.zeros(len(predicted_items), dtype=np.float64)
    for real, pred in zip(real_items, predicted_items):

        real = real[real > 0]
        pred = pred[pred > 0]

        real_found_in_pred = np.isin(pred, real, assume_unique=True)

        if real_found_in_pred.any():
            recommended = real_found_in_pred.sum()
            recall = recommended / len(real)
        else:
            recall = 0

        recalls[idx] += recall
        idx += 1
        return recalls.mean()
