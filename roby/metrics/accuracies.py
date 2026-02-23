import numpy as np
from .utility import get_correct_predictions

def compute_trad_accuracy(predictions, true_labels):
    '''Calcola l'accuratezza tradizionale.'''
    correct_predictions = get_correct_predictions(predictions, true_labels)
    return correct_predictions.mean()

def compute_thresholded_accuracy(predictions, uncertainties, true_labels, threshold):
    '''Calcola l'accuratezza considerando solo le predizioni con incertezza < threshold.'''
    correct = get_correct_predictions(predictions, true_labels)
    mask = uncertainties < threshold
    if not np.any(mask):
        return 0.0
    return correct[mask].mean()

def compute_unc_weighted_accuracy(predictions, uncertainties, true_labels):
    '''Calcola l'accuratezza pesata: le corrette pesano (1-u), le errate pesano u.'''
    correct = get_correct_predictions(predictions, true_labels).astype(np.float64)
    u = uncertainties.astype(np.float64)
    # Formula: Corrette * Certezza + Errate * Incertezza (penalità minore se incerto su errore)
    # Nota: La formula originale della repo era: correct*(1-u) + (1-correct)*u
    return np.mean(correct * (1.0 - u) + (1.0 - correct) * u)

def compute_net_cert_accuracy(predictions, uncertainties, true_labels):
    '''Calcola la Net Certainty Accuracy: (Certezza su Corrette) - (Certezza su Errate).'''
    correct = get_correct_predictions(predictions, true_labels).astype(np.float64)
    cert = 1.0 - uncertainties.astype(np.float64)
    return np.mean((correct * cert) - ((1 - correct) * cert))

def compute_accuracies(predictions, uncertainties, true_labels, threshold):
    '''Calcola e restituisce tutte e 4 le metriche di accuratezza.'''
    return (
        compute_trad_accuracy(predictions, true_labels),
        compute_thresholded_accuracy(predictions, uncertainties, true_labels, threshold),
        compute_unc_weighted_accuracy(predictions, uncertainties, true_labels),
        compute_net_cert_accuracy(predictions, uncertainties, true_labels),
    )