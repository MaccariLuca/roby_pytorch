import numpy as np
from roby import EnvironmentRTest, Alterations, Utility
from roby.ReferralResults import ReferralResults
from roby.uncertainty import measures as unc_measures

def referral_test(environment: EnvironmentRTest.EnvironmentRTest,
                  alteration: Alterations.Alteration = None,
                  alteration_level: float = 0.0,
                  unc_method: str = 'entropy',
                  n_mc_samples: int = 10,
                  referral_steps: int = 20,
                  doctor_acc: float = 1.0,    # Accuratezza del Dottore
                  cost_fn: float = 500.0,     # Costo Falso Negativo (Grave)
                  cost_fp: float = 100.0,     # Costo Falso Positivo
                  cost_doctor: float = 10.0,  # Costo riferimento medico
                  cost_ai: float = 0.0,        # Costo inferenza IA
                  time_ai=0.1,             # Tempo inferenza IA (sec)
                  time_doc_base=30.0,       # Tempo base medico (sec)
                  time_doc_weight=1.0      # Moltiplicatore complessità (0 = tempo fisso)
                  ) -> ReferralResults:
    """
    Esegue il test di Referral Learning con chunking per la memoria e analisi costi FN/FP.
    """

    # 1. Caricamento dati
    print("Caricamento dataset in memoria...")
    x_raw, y_true = Utility.load_raw_data_multicore(environment)
    total_samples = len(y_true)
    num_pos = int(np.sum(y_true == 1))
    num_neg = int(np.sum(y_true == 0))
    
    # 2. Pipeline: Alterazione + Preprocessing
    msg = f" (Alt: {alteration.name()} lvl {alteration_level})" if (alteration and alteration_level > 0) else ""
    print(f"Elaborazione immagini{msg}...")
    x_data = Utility.process_batch_multicore(
        x_raw, alteration, alteration_level, environment.pre_processing
    )
    
    # 3. Calcolo Incertezza con Chunking (Evita OOM)
    print(f"Calcolo incertezza ({unc_method})...")
    chunk_size = 32
    all_preds = []
    all_uncertainties = []

    is_stochastic = unc_method in ['mi', 'mean_entropy', 'mean_var']

    for i in range(0, total_samples, chunk_size):
        x_chunk = x_data[i : i + chunk_size]
        
        if is_stochastic:
            if environment.ensemble_models:
                chunk_mc_preds = np.array([m.predict(x_chunk, verbose=0) for m in environment.ensemble_models])
            else:
                chunk_mc_preds = np.array([environment.model(x_chunk, training=True).numpy() for _ in range(n_mc_samples)])
            
            chunk_preds = np.mean(chunk_mc_preds, axis=0)
            if unc_method == 'mi': chunk_unc = unc_measures.mutual_information(chunk_mc_preds)
            elif unc_method == 'mean_entropy': chunk_unc = unc_measures.mean_entropy(chunk_mc_preds)
            else: chunk_unc = unc_measures.mean_predictive_variance(chunk_mc_preds)
        else:
            if environment.ensemble_models:
                chunk_preds = np.mean(np.array([m.predict(x_chunk, verbose=0) for m in environment.ensemble_models]), axis=0)
            else:
                chunk_preds = environment.model.predict(x_chunk, verbose=0)
            
            if unc_method == 'confidence': chunk_unc = unc_measures.confidence_complement(chunk_preds)
            else: chunk_unc = unc_measures.entropy(chunk_preds)
        
        all_preds.append(chunk_preds)
        all_uncertainties.append(chunk_unc)
        print(f" > Progresso: {min(i + chunk_size, total_samples)}/{total_samples}", end='\r')

    predictions = np.vstack(all_preds)
    uncertainties = np.concatenate(all_uncertainties)

    u_min, u_max = np.min(uncertainties), np.max(uncertainties)
    norm_u = (uncertainties - u_min) / (u_max - u_min) if u_max > u_min else np.zeros_like(uncertainties)
    print("\nIncertezza calcolata.")

    # 4. Preparazione maschere e ordinamenti
    pred_classes = np.argmax(predictions, axis=1)
    ai_correct_mask = (pred_classes == y_true)
    fn_mask_ai = (y_true == 1) & (pred_classes == 0)
    fp_mask_ai = (y_true == 0) & (pred_classes == 1)
    
    np.random.seed(42) 
    doc_correct_mask = np.random.rand(total_samples) < doctor_acc
    
    sorted_indices = np.argsort(uncertainties) 
    oracle_scores = (~ai_correct_mask).astype(float) + np.random.uniform(0, 1e-6, size=total_samples)
    sorted_indices_oracle = np.argsort(oracle_scores) 

    # --- Calcolo Tempi Medici ---
    # Tempo con difficoltà variabile
    doc_times_weighted = time_doc_base * (1 + time_doc_weight * norm_u)
    avg_time_doc_weighted = np.mean(doc_times_weighted)
    # Tempo fisso
    avg_time_doc_fixed = time_doc_base

    # Inizializzazione liste
    referral_rates = np.linspace(0, 0.99, referral_steps)
    model_accuracies, system_accuracies, oracle_accuracies, total_costs, thresholds = [], [], [], [], []
    referral_times_weighted = []
    referral_times_fixed = []

    # 5. Simulazione Referral
    print(f"Simulazione Scenario Ibrido...")
    for rate in referral_rates:
        n_refer = int(total_samples * rate)
        n_keep = total_samples - n_refer
        
        indices_keep = sorted_indices[:n_keep] if n_keep > 0 else []
        indices_refer = sorted_indices[n_keep:] if n_refer > 0 else []
        
        # Performance
        acc_model = np.mean(ai_correct_mask[indices_keep]) if n_keep > 0 else 0.0
        n_correct_ai_kept = np.sum(ai_correct_mask[indices_keep]) if n_keep > 0 else 0
        n_correct_doc_ref = np.sum(doc_correct_mask[indices_refer]) if n_refer > 0 else 0
        acc_system = (n_correct_ai_kept + n_correct_doc_ref) / total_samples

        # Tempi Sistema Ibrido (Weighted vs Fixed)
        time_hybrid_weighted = (total_samples * time_ai) + np.sum(doc_times_weighted[indices_refer])
        time_hybrid_fixed = (total_samples * time_ai) + (n_refer * time_doc_base)
        
        referral_times_weighted.append(time_hybrid_weighted / total_samples)
        referral_times_fixed.append(time_hybrid_fixed / total_samples)

        # Costi
        n_fn_ai = np.sum(fn_mask_ai[indices_keep]) if n_keep > 0 else 0
        n_fp_ai = np.sum(fp_mask_ai[indices_keep]) if n_keep > 0 else 0
        if n_refer > 0:
            doc_err_indices = indices_refer[~doc_correct_mask[indices_refer]]
            n_fn_doc = np.sum(y_true[doc_err_indices] == 1)
            n_fp_doc = np.sum(y_true[doc_err_indices] == 0)
        else:
            n_fn_doc = n_fp_doc = 0

        c_errors = ((n_fn_ai + n_fn_doc) * cost_fn) + ((n_fp_ai + n_fp_doc) * cost_fp)
        total_costs.append((total_samples * cost_ai) + (n_refer * cost_doctor) + c_errors)
        
        oracle_accuracies.append(np.mean(ai_correct_mask[sorted_indices_oracle[:n_keep]]) if n_keep > 0 else 0.0)
        model_accuracies.append(acc_model)
        system_accuracies.append(acc_system)
        thresholds.append(uncertainties[indices_keep[-1]] if n_keep > 0 else -1.0)

    # 6. Analisi Ottimo
    min_cost_idx = np.argmin(total_costs)
    opt_rate = referral_rates[min_cost_idx]
    n_refer_opt = int(total_samples * opt_rate)
    indices_refer_opt = sorted_indices[total_samples - n_refer_opt:] if n_refer_opt > 0 else []

    # KPI Qualità
    total_ai_fn, total_ai_fp = np.sum(fn_mask_ai), np.sum(fp_mask_ai)
    total_errors = total_ai_fn + total_ai_fp
    captured = np.sum(fn_mask_ai[indices_refer_opt]) + np.sum(fp_mask_ai[indices_refer_opt])
    
    err_ref_pct = captured / total_errors if total_errors > 0 else 0
    fn_captured = np.sum(fn_mask_ai[indices_refer_opt]) / total_ai_fn if total_ai_fn > 0 else 0
    fp_captured = np.sum(fp_mask_ai[indices_refer_opt]) / total_ai_fp if total_ai_fp > 0 else 0
    correct_retained = np.sum(ai_correct_mask[sorted_indices[:total_samples-n_refer_opt]]) / np.sum(ai_correct_mask)

    # AUSE
    diffs = np.array(oracle_accuracies) - np.array(model_accuracies)
    try: ause = np.trapezoid(diffs, referral_rates)
    except AttributeError: ause = np.trapz(diffs, referral_rates)

    return ReferralResults(
        referral_rates=list(referral_rates),
        model_accuracies=model_accuracies,
        system_accuracies=system_accuracies,
        oracle_accuracies=oracle_accuracies,
        referral_times=referral_times_weighted, # Main curve
        baseline_times={'ai': time_ai, 'doc': avg_time_doc_weighted, 'doc_fixed': avg_time_doc_fixed},
        total_costs=total_costs,
        ause=float(ause),
        thresholds=thresholds,
        title=f"Referral Analysis",
        unc_method=unc_method,
        simulation_params=None, # Gestito esternamente
        extra_metrics={
            'err_ref_pct': float(err_ref_pct),
            'fn_ref_pct': float(fn_captured),
            'fp_ref_pct': float(fp_captured),
            'correct_retained_pct': float(correct_retained),
            'referral_times_fixed': referral_times_fixed
        }
    )