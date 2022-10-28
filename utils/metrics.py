from scipy.stats import spearmanr
from sklearn import metrics
from torch import Tensor


def cls_micro_metrics(preds, labels, desc='') -> dict:
    return {
        f'val/acc{desc}': metrics.accuracy_score(labels, preds),
        f'val/precision{desc}': metrics.precision_score(labels, preds, average='micro', zero_division=0),
        f'val/recall{desc}': metrics.recall_score(labels, preds, average='micro', zero_division=0),
        f'val/f1{desc}': metrics.f1_score(labels, preds, average='micro', zero_division=0)
    }


def cls_macro_metrics(preds, labels, desc='') -> dict:
    return {
        f'val/acc{desc}': metrics.accuracy_score(labels, preds),
        f'val/precision{desc}': metrics.precision_score(labels, preds, average='macro', zero_division=0),
        f'val/recall{desc}': metrics.recall_score(labels, preds, average='macro', zero_division=0),
        f'val/f1{desc}': metrics.f1_score(labels, preds, average='macro', zero_division=0)
    }


def top_k_accuracy_score(preds_score, labels, k):
    if isinstance(preds_score, Tensor):
        preds_score = preds_score.tolist()
        labels = labels.tolist()
    return {f'val/accuracy_top_{k}': metrics.top_k_accuracy_score(labels, preds_score, k=k)}


def top_k_accuracy(preds_idx: list, labels: list, k):
    count = 0
    for p, l in zip(preds_idx, labels):
        if l in p[:k]:
            count += 1
    return {f'val/accuracy_top_{k}': count / len(preds_idx)}


def spearman_corr(preds, labels):
    if isinstance(preds, Tensor):
        preds = preds.tolist()
        labels = labels.tolist()
    spearman_corr = spearmanr(preds, labels).correlation
    return {'val/spearman_corr': spearman_corr}


def cls_report(preds, labels):
    return metrics.classification_report(labels, preds, zero_division=0)
