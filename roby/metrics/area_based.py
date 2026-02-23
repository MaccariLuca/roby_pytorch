import numpy as np
from .utility import get_correct_predictions

def compute_certainty_data(predictions, uncertainties, true_labels):
    '''Separa e ordina le certezze per predizioni corrette ed errate.'''
    correct_mask = get_correct_predictions(predictions, true_labels).astype(bool)
    certainties = 1.0 - uncertainties
    correct_certainties = np.sort(certainties[correct_mask])
    incorrect_certainties = np.sort(certainties[~correct_mask])
    return correct_certainties, incorrect_certainties

def _compute_certainty_areas(correct_certainties, incorrect_certainties):
    '''Calcola le aree sotto le curve di certezza (Somma di Riemann).'''
    n_incorrect = len(incorrect_certainties)
    au_correct = correct_certainties.sum()
    au_incorrect = incorrect_certainties.sum()
    aa_incorrect = n_incorrect - au_incorrect
    return au_correct, au_incorrect, aa_incorrect

def compute_acua(correct_certainties, incorrect_certainties):
    '''Calcola ACUA (Aggregate Certainty–Uncertainty Area).'''
    total_area = len(correct_certainties) + len(incorrect_certainties)
    if total_area == 0: return 0.0
    au_correct, _, aa_incorrect = _compute_certainty_areas(correct_certainties, incorrect_certainties)
    return (au_correct + aa_incorrect) / total_area

def compute_nca(correct_certainties, incorrect_certainties):
    '''Calcola NCA (Net Certainty Area).'''
    total_area = len(correct_certainties) + len(incorrect_certainties)
    if total_area == 0: return 0.0
    au_correct, au_incorrect, _ = _compute_certainty_areas(correct_certainties, incorrect_certainties)
    return (au_correct - au_incorrect) / total_area

def compute_area_based_metrics(y_true, y_pred_classes, uncertainties, num_classes):
    '''Wrapper per calcolare ACUA e NCA dato l'input standard.'''
    # Ricostruiamo predictions "fittizie" o usiamo utility in modo intelligente
    # Qui simuliamo l'uso di utility passando array compatibili
    # Nota: per semplicità ricalcoliamo le mask qui se necessario o adattiamo gli input
    # Per coerenza con lo stile "roby", usiamo direttamente le funzioni sopra se abbiamo già i dati
    # Ma per questo wrapper specifico:
    correct_mask = (y_pred_classes == np.argmax(y_true, axis=1)) if y_true.ndim > 1 else (y_pred_classes == y_true)
    certainties = 1.0 - uncertainties
    c_cert = np.sort(certainties[correct_mask])
    i_cert = np.sort(certainties[~correct_mask])
    return compute_acua(c_cert, i_cert), compute_nca(c_cert, i_cert)