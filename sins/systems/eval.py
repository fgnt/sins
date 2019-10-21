import numpy as np


def true_false_positives_negatives(
        target_mat, decision_mat, sum_axis=None
):
    """

    Args:
        target_mat: n_hot matrix indicating ground truth events/labels
            (num_frames times num_labels)
        decision_mat: n_hot matrix indicating detected events/labels
            (num_frames times num_labels)
        sum_axis:

    Returns:

    """
    tp = target_mat * decision_mat
    fp = (1. - target_mat) * decision_mat
    tn = (1. - target_mat) * (1. - decision_mat)
    fn = target_mat * (1. - decision_mat)
    if sum_axis is not None:
        tp = np.sum(tp, axis=sum_axis)
        fp = np.sum(fp, axis=sum_axis)
        tn = np.sum(tn, axis=sum_axis)
        fn = np.sum(fn, axis=sum_axis)
    return tp, fp, tn, fn


def fscore(target_mat, decision_mat, beta=1., event_wise=False):
    """

    Args:
        target_mat: n_hot matrix indicating ground truth events/labels
            (num_frames times num_labels)
        decision_mat: n_hot matrix indicating detected events/labels
            (num_frames times num_labels)
        event_wise:
        beta:

    Returns:

    """
    sum_axis = -2 if event_wise else (-2, -1)
    tp, fp, tn, fn = true_false_positives_negatives(
        target_mat, decision_mat, sum_axis
    )
    n_sys = np.maximum(np.sum(decision_mat, axis=sum_axis), 1e-15)
    n_ref = np.maximum(np.sum(target_mat, axis=sum_axis), 1e-15)
    precision = tp / n_sys
    recall = tp / n_ref
    f_beta = (1 + beta**2) * precision * recall / np.maximum(
        beta**2 * precision + recall, 1e-15)
    return f_beta, precision, recall
