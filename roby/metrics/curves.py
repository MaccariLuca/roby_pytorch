import numpy as np
from .utility import get_correct_predictions

def compute_uroc(predictions, uncertainties, true_labels, n_thresholds=100):
    '''Calcola i punti della curva Uncertainty ROC.'''
    correct = get_correct_predictions(predictions, true_labels).astype(bool)
    num_correct = int(correct.sum())
    num_incorrect = int(correct.size - num_correct)
    
    if num_correct == 0 or num_incorrect == 0:
        return None, None
        
    quantiles = np.linspace(1.0, 0.0, num=n_thresholds+1)
    thresholds = np.quantile(uncertainties, quantiles)
    points = []
    
    for t in thresholds:
        unknown = uncertainties > t
        unknown_on_correct = int((unknown & correct).sum())
        unknown_on_incorrect = int((unknown & ~correct).sum())
        x = unknown_on_correct / num_correct
        y = unknown_on_incorrect / num_incorrect
        points.append((x, y))
    return points, thresholds

def compute_auc(points):
    '''Calcola l'area sotto la curva (trapezoidale).'''
    if points is None:
        return 0.0 # Return 0.0 instead of None for safety
    x, y = map(np.asarray, zip(*points))
    return float(np.trapezoid(y, x))