import numpy as np
from datetime import datetime
from roby import EnvironmentRTest, Alterations
from roby.ReferralResults import ReferralResults
from roby.RobustnessResults import RobustnessResults
from roby.uncertainty import measures as unc_measures
from roby.metrics import accuracies as acc_metrics

def _load_raw_data(environment):
    """
    Carica i dati grezzi (senza preprocessing) per poter applicare le alterazioni correttamente.
    Restituisce una lista di immagini e un array di label.
    """
    x_list = []
    y_list = []
    
    if not environment.file_list:
        return [], np.array([])

    for i, f in enumerate(environment.file_list):
        # 1. Load Image
        if isinstance(f, str):
            if environment.reader_f: 
                img = environment.reader_f(f)
            else: 
                import cv2
                img = cv2.imread(f)
        else: 
            img = f
        
        # IMPORTANTE: Non applichiamo ancora il preprocessing qui!
        x_list.append(img)
        
        # 2. Labels
        if environment.label_list:
            lbl = environment.label_list[i]
            try: 
                idx = environment.classes.index(str(lbl))
            except ValueError: 
                idx = -1
            y_list.append(idx)
            
    return x_list, np.array(y_list)

def _process_batch(x_list, environment, alteration=None, level=0.0):
    """
    Applica alterazione (opzionale) e POI il preprocessing alla lista di immagini.
    Restituisce un array numpy batch (N, H, W, C).
    """
    processed_imgs = []
    for img in x_list:
        # 1. Alteration (Su immagine grezza)
        if alteration and level > 0:
            img = alteration.apply_alteration(img, level)
        
        # 2. Preprocessing (Resize, Normalize, ExpandDims)
        if environment.pre_processing:
            img = environment.pre_processing(img)
            
        processed_imgs.append(img)
        
    if not processed_imgs:
        return np.array([])
        
    return np.vstack(processed_imgs)

def robustness_test(environment, alteration, npoints=10, theta=0.5):
    """
    Esegue il test di robustezza classico richiesto dal sistema.
    """
    print(f"--- ROBUSTEZZA: {alteration.name()} ---")
    x_raw, y_true = _load_raw_data(environment)
    
    levels = np.linspace(0, 1, npoints)
    accuracies = []

    for lvl in levels:
        x_batch = _process_batch(x_raw, environment, alteration, lvl)
        preds = environment.model.predict(x_batch, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        acc = np.mean(pred_labels == y_true)
        accuracies.append(acc)
        robustness_value = float(np.mean(accuracies))
        print(f"  Livello {lvl:.2f}: Accuratezza {acc:.4f}")

    return RobustnessResults(
        steps=levels.tolist(),              
        accuracies=accuracies,
        robustness=robustness_value,         
        title=f"Robustness Test: {alteration.name()}",
        xlabel="Alteration Level",           
        ylabel="Accuracy",               
        alteration_name=alteration.name(),
        threshold=theta                    
    )

def referral_test(environment: EnvironmentRTest.EnvironmentRTest,
                  alteration: Alterations.Alteration = None,
                  alteration_level: float = 0.0,
                  unc_method: str = 'entropy',
                  n_mc_samples: int = 10,
                  referral_steps: int = 20) -> ReferralResults:
    
    # 0. Caricamento Dati Grezzi
    print("Caricamento dataset in memoria (Raw)...")
    x_raw, y_true = _load_raw_data(environment)
    
    # 1. Preparazione Batch (Alterazione + Preprocessing)
    msg = f" (Alt: {alteration.name()} lvl {alteration_level})" if (alteration and alteration_level > 0) else ""
    print(f"Elaborazione immagini{msg}...")
    x_data = _process_batch(x_raw, environment, alteration, alteration_level)
    
    # 2. Calcolo Incertezza
    print(f"Calcolo incertezza ({unc_method})...")
    
    if unc_method in ['mi', 'mean_entropy', 'mean_var']:
        # MC Dropout / Ensemble
        if environment.ensemble_models:
            print(f" -> Ensemble ({len(environment.ensemble_models)} models)")
            preds_list = [m.predict(x_data, verbose=0) for m in environment.ensemble_models]
            mc_preds = np.array(preds_list)
        else:
            # Single Model MC Dropout
            print(f" -> MC Dropout ({n_mc_samples} samples)")
            mc_preds = np.array([environment.model(x_data, training=True).numpy() for _ in range(n_mc_samples)])
            
        predictions = np.mean(mc_preds, axis=0)
        
        if unc_method == 'mi': uncertainties = unc_measures.mutual_information(mc_preds)
        elif unc_method == 'mean_entropy': uncertainties = unc_measures.mean_entropy(mc_preds)
        elif unc_method == 'mean_var': uncertainties = unc_measures.mean_predictive_variance(mc_preds)
        else: uncertainties = unc_measures.entropy(predictions)

    else:
        # Standard Single Model (No MC)
        if environment.ensemble_models:
             print(f" -> Ensemble Mean (Standard {unc_method})")
             preds_list = [m.predict(x_data, verbose=0) for m in environment.ensemble_models]
             predictions = np.mean(np.array(preds_list), axis=0)
        else:
             predictions = environment.model.predict(x_data, verbose=0)
             
        if unc_method == 'confidence':
            uncertainties = unc_measures.confidence_complement(predictions)
        else:
            uncertainties = unc_measures.entropy(predictions)

    # 3. Ground Truth
    pred_classes = np.argmax(predictions, axis=1)
    correct_preds = (pred_classes == y_true)
    
    # 4. Curve Referral & Oracle
    referral_rates = np.linspace(0, 0.95, referral_steps)
    model_accuracies = []
    oracle_accuracies = []
    thresholds_used = []
    
    print("Calcolo curve...")
    sorted_indices_model = np.argsort(uncertainties) 
    
    # Oracle: Score alto (1.0) per Errati, Basso (0.0) per Corretti.
    oracle_scores = (~correct_preds).astype(float) 
    oracle_scores += np.random.uniform(0, 1e-6, size=len(oracle_scores)) 
    sorted_indices_oracle = np.argsort(oracle_scores) 

    total_samples = len(y_true)
    
    for rate in referral_rates:
        n_drop = int(total_samples * rate)
        n_keep = total_samples - n_drop
        
        # Model Curve
        if n_keep > 0:
            keep_indices_m = sorted_indices_model[:n_keep]
            acc_model = np.mean(correct_preds[keep_indices_m])
            thresh = uncertainties[sorted_indices_model[n_keep-1]]
        else:
            acc_model = 0.0
            thresh = np.max(uncertainties)
            
        model_accuracies.append(acc_model)
        thresholds_used.append(thresh)

        # Oracle Curve
        if n_keep > 0:
            keep_indices_o = sorted_indices_oracle[:n_keep]
            acc_oracle = np.mean(correct_preds[keep_indices_o])
        else:
            acc_oracle = 0.0
        oracle_accuracies.append(acc_oracle)

    # 5. AUSE
    diffs = np.array(oracle_accuracies) - np.array(model_accuracies)
    
    # FIX PER NUMPY 2.0+: uso np.trapezoid invece di np.trapz
    try:
        ause = np.trapezoid(diffs, referral_rates)
    except AttributeError:
        ause = np.trapz(diffs, referral_rates)

    alt_name = alteration.name() if (alteration and alteration_level > 0) else "Original Data"
    
    return ReferralResults(
        referral_rates=list(referral_rates),
        accuracies=model_accuracies,
        oracle_accuracies=oracle_accuracies,
        ause=float(ause),
        thresholds=thresholds_used,
        title=f"Referral on {alt_name}",
        unc_method=unc_method
    )

def display_robustness_results(results):
    """Visualizza un sommario testuale dei risultati di robustezza."""
    print(f"\n--- Risultati Robustezza per: {results.alteration_name} ---")
    print(f"{'Livello':<10} {'Accuratezza':<10}")
    for i, step in enumerate(results.steps):
        print(f"{step:<10.2f} {results.accuracies[i]:<10.4f}")
    print("-" * 30)
    print(f"Robustness Score: {results.robustness:.4f}")
    print("-" * 30)