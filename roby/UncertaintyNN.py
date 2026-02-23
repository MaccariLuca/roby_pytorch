import numpy as np
import tensorflow as tf
from datetime import datetime

from roby import EnvironmentRTest, Alterations
from roby.UncertaintyResults import UncertaintyResults

# Import dai nuovi pacchetti modulari
from roby.uncertainty import measures as unc_measures
from roby.metrics import accuracies as acc_metrics
from roby.metrics import area_based as area_metrics
from roby.metrics import curves as curve_metrics
from roby.metrics import calibration as cal_metrics
from roby.viz import robustness as viz_rob

def _load_raw_data(environment):
    """Carica immagini grezze per permettere alterazione corretta."""
    x_list = []
    y_list = []
    
    if not environment.file_list: return [], np.array([])
    
    for i, f in enumerate(environment.file_list):
        if isinstance(f, str):
            if environment.reader_f: img = environment.reader_f(f)
            else: import cv2; img = cv2.imread(f)
        else: img = f
        x_list.append(img)
        
        if environment.label_list:
            lbl = environment.label_list[i]
            try: idx = environment.classes.index(str(lbl))
            except ValueError: idx = -1
            if idx == -1:
                print(f"ATTENZIONE: Label '{lbl}' non trovata in classes.csv!")
            y_list.append(idx)
    return x_list, np.array(y_list)

def _process_batch(x_list, environment, alteration, level):
    """Applica alterazione e poi preprocessing."""
    processed = []
    for img in x_list:
        if alteration and level > 0:
            img = alteration.apply_alteration(img, level)
        if environment.pre_processing:
            img = environment.pre_processing(img)
        processed.append(img)
    if not processed: return np.array([])
    return np.vstack(processed)

def uncertainty_test(environment: EnvironmentRTest.EnvironmentRTest,
                     alteration: Alterations.Alteration,
                     n_values: int,
                     uncertainty_method='entropy',
                     threshold=0.5,
                     n_mc_samples=10) -> UncertaintyResults:
    
    # 0. Caricamento Dati Grezzi
    x_raw, y_test = _load_raw_data(environment)
    num_classes = len(environment.classes)

    steps = []
    res_acc = []
    res_th_acc = []
    res_w_acc = []
    res_au_uroc = []
    res_acua = []
    res_nca = []
    res_ece = []
    times = []
    
    mcd_methods = ['mi', 'mean_entropy', 'mean_var']
    is_mcd = uncertainty_method in mcd_methods
    
    print(f"Avvio test incertezza: {alteration.name()} (Method: {uncertainty_method})")
    
    for i in range(n_values):
        val = i / (n_values - 1) if n_values > 1 else 0
        steps.append(val)
        
        start_time = datetime.now()
        
        # 1. Pipeline: Raw -> Alter -> Preprocess
        x_batch = _process_batch(x_raw, environment, alteration, val)
        
        # 2. Inferenza
        if is_mcd:
            if environment.ensemble_models:
                 mc_preds = np.array([m.predict(x_batch, verbose=0) for m in environment.ensemble_models])
            else:
                 mc_preds = np.array([environment.model(x_batch, training=True).numpy() for _ in range(n_mc_samples)])
                 
            predictions = np.mean(mc_preds, axis=0)
            
            if uncertainty_method == 'mi': uncertainties = unc_measures.mutual_information(mc_preds)
            elif uncertainty_method == 'mean_entropy': uncertainties = unc_measures.mean_entropy(mc_preds)
            elif uncertainty_method == 'mean_var': uncertainties = unc_measures.mean_predictive_variance(mc_preds)
            else: uncertainties = unc_measures.mean_entropy(mc_preds)
        else:
            if environment.ensemble_models:
                preds_list = [m.predict(x_batch, verbose=0) for m in environment.ensemble_models]
                predictions = np.mean(np.array(preds_list), axis=0)
            else:
                predictions = environment.model.predict(x_batch, verbose=0)
            
            if uncertainty_method == 'confidence':
                uncertainties = unc_measures.confidence_complement(predictions)
            else:
                uncertainties = unc_measures.entropy(predictions)

        # 3. Metriche
        acc, th_acc, w_acc, nca = acc_metrics.compute_accuracies(
            predictions, uncertainties, y_test, threshold
        )
        
        points, _ = curve_metrics.compute_uroc(predictions, uncertainties, y_test)
        au_uroc = curve_metrics.compute_auc(points)
        
        y_pred_classes = np.argmax(predictions, axis=1)
        acua, _ = area_metrics.compute_area_based_metrics(
            y_test, y_pred_classes, uncertainties, num_classes
        )

        ece = cal_metrics.compute_ece(predictions, y_test)
        
        end_time = datetime.now()
        times.append((end_time - start_time).total_seconds())
        
        res_acc.append(acc)
        res_th_acc.append(th_acc)
        res_w_acc.append(w_acc)
        res_nca.append(nca)
        res_au_uroc.append(au_uroc)
        res_acua.append(acua)
        res_ece.append(ece)
        
        if (i+1) % 5 == 0 or i == 0:
             print(f"Step {i+1}/{n_values} - Acc: {acc:.2f} | ECE: {ece:.2f} | AU-UROC: {au_uroc:.2f}")

    results = UncertaintyResults(
        steps=steps,
        accuracies=res_acc,
        thresholded_accuracies=res_th_acc,
        weighted_accuracies=res_w_acc,
        au_urocs=res_au_uroc,
        acuas=res_acua,
        ncas=res_nca,
        eces=res_ece,
        title=f"Uncertainty ({uncertainty_method}) over {alteration.name()}",
        alteration_name=alteration.name(),
        times=times
    )
    
    viz_rob.plot_robustness_results(results, threshold=threshold)
    return results