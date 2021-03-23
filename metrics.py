from collections import defaultdict
from sklearn import metrics
import numpy as np
import keras.backend as K


def ap_score(cands):
    """
    cands: (predicted_scores, actual_labels)
    Using: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

    It uses roc-auc and then computes avg-precision.
    """
    count = 0
    score = 0
    for i, (y_true, _) in enumerate(cands):
        if y_true > 0:
            count += 1.0
            precision_q_i = count / (i + 1.0)
            score += precision_q_i

    if count == 0:
        return None

    return score / (count + 1e-6)


def map_score(qids, labels, preds):
    """
    Method that computes Mean Average Precision for the given input.
    Authors use their custom method to train. Actual benchmark is done using TREC eval.
    Original Code:
    https://github.com/aseveryn/deep-qa/blob/master/dep.py#L403
    Read more about it:
    https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/metrics/ranking.py#L107
    https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/
    http://fastml.com/what-you-wanted-to-know-about-mean-average-precision/
    """
    qid_2_cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        assert all([pred >= 0, pred <= 1])
        qid_2_cand[qid].append((label, pred))

    avg_precisions = []
    for qid, cands in qid_2_cand.items():
        # get average prec score for all cands of qid
        avg_precision = ap_score(sorted(cands, reverse=True, key=lambda x: x[1]))
        if avg_precision is None:
            continue

        avg_precisions.append(avg_precision)

    return sum(avg_precisions) / len(avg_precisions)


def map_score_2(qids, labels, preds):
    qid_2_cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        assert all([pred >= 0, pred <= 1])
        qid_2_cand[qid].append((label, pred))

    avg_precisions = []
    for qid, cands in qid_2_cand.items():
        cands = sorted(cands, reverse=True, key=lambda x: x[1])
        y_true, y_pred = map(list, zip(*cands))
        avg_precision = metrics.average_precision_score(y_true, y_pred)
        if not np.isnan(avg_precision):
            avg_precisions.append(avg_precision)

    return sum(avg_precisions) / len(avg_precisions)


def precision_at_k(qids, labels, preds, k=30):
    qid_2_cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        assert all([pred >= 0, pred <= 1])
        qid_2_cand[qid].append((label, pred))

    precisions_at_k = []
    for qid, cands in qid_2_cand.items():
        cands = sorted(cands, reverse=True, key=lambda x: x[1])[0:k]
        precision = 0
        for i, (y_true, _) in enumerate(cands):
            if y_true > 0:
                precision += 1.0
        precision_at_k = precision / k
        precisions_at_k.append(precision_at_k)

    return sum(precisions_at_k) / len(precisions_at_k)



def precision(y_true, y_pred):
    """
    Precision metric.	
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of	
    how many selected items are relevant.	
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_score = true_positives / (predicted_positives + K.epsilon())
    return precision_score


def recall(y_true, y_pred):
    """Recall metric.	
        Only computes a batch-wise average of recall.	
        Computes the recall, a metric for multi-label classification of	
        how many relevant items are selected.	
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_score = true_positives / (possible_positives + K.epsilon())

    return recall_score

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))
