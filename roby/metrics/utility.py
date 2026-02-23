import numpy as np

def get_correct_predictions(predictions, true_labels):
    '''
    Restituisce array booleano: 1 se la predizione è corretta, 0 altrimenti.
    Gestisce sia one-hot encoding che sparse labels.
    '''
    pred_classes = np.argmax(predictions, axis=1)
    
    # Se true_labels è one-hot (2D), lo convertiamo in classi (1D)
    if true_labels.ndim > 1:
        true_classes = np.argmax(true_labels, axis=1)
    else:
        true_classes = true_labels
        
    return (pred_classes == true_classes).astype(np.int32)