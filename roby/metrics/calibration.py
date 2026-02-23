import numpy as np
import math

def compute_ece(predictions, true_labels, n_bins=10):
    '''
    Calcola l'Expected Calibration Error (ECE).
    Misura quanto la confidenza del modello è allineata con la sua accuratezza reale.
    '''
    bins = np.linspace(start=0.0, stop=1.0, num=n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        start, stop = bins[i], bins[i + 1]
        in_bin_correct, in_bin_conf = [], []
        for pred, true_label in zip(predictions, true_labels):
            conf = np.max(pred)
            # Gestione ultimo bin inclusivo
            in_last_bin = (i + 1 == n_bins) and math.isclose(conf, stop)
            
            if (start <= conf < stop) or in_last_bin:
                correct = 1 if np.argmax(pred) == true_label else 0
                in_bin_correct.append(correct)
                in_bin_conf.append(conf)
                
        if len(in_bin_conf) > 0:
            mean_acc = np.mean(in_bin_correct)
            mean_conf = np.mean(in_bin_conf)
            weight = len(in_bin_conf) / len(predictions)
            ece += weight * abs(mean_acc - mean_conf)
    return ece