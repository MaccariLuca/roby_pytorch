import numpy as np
from enum import Enum

class Measure(Enum):
    CONF_COMPLEMENT = "conf_compl"
    ENTROPY = "entropy"
    MEAN_ENTROPY = "mean_entropy"
    MI = "mi"
    MEAN_VAR = "mean_var"
    MAX_VAR = "max_var"
    EKL = "ekl"

def confidence_complement(predictions):
    '''1 - Confidence'''
    preds = np.asarray(predictions, dtype=np.float64)
    return 1.0 - np.max(preds, axis=1)

def entropy(predictions):
    '''Entropia di Shannon'''
    EPS = 1e-12
    preds = np.asarray(predictions, dtype=np.float64)
    preds = np.clip(preds, EPS, 1.0)
    return -np.sum(preds * np.log(preds), axis=1)

def mean_entropy(predictions):
    """Entropia della media (per Ensemble/MCD)"""
    mean_preds = np.mean(np.asarray(predictions, dtype=np.float64), axis=0)
    return entropy(mean_preds)

def mutual_information(predictions):
    '''Mutual Information (Incertezza Epistemica)'''
    preds = np.asarray(predictions, dtype=np.float64)
    # H(Mean)
    H_mean = entropy(preds.mean(axis=0))
    # Mean(H)
    EPS = 1e-12
    preds_clipped = np.clip(preds, EPS, 1.0)
    H_each = -np.sum(preds_clipped * np.log(preds_clipped), axis=2)
    mean_H = H_each.mean(axis=0)
    return H_mean - mean_H

def _get_normalized_variance(predictions):
    preds = np.asarray(predictions, dtype=np.float64)
    var = np.var(preds, axis=0)
    return var

def mean_predictive_variance(predictions):
    return _get_normalized_variance(predictions).mean(axis=1)

def max_predictive_variance(predictions):
    return _get_normalized_variance(predictions).max(axis=1)

def expcected_kl(predictions):
    preds = np.asarray(predictions, dtype=np.float64)
    EPS = 1e-12
    preds = np.clip(preds, EPS, 1.0)
    mean_preds = np.broadcast_to(preds.mean(axis=0), preds.shape)
    log_mean = np.log(mean_preds)
    log_preds = np.log(preds)
    kl = np.sum(mean_preds * (log_mean - log_preds), axis=2)
    return np.mean(kl, axis=0)