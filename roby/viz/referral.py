import matplotlib.pyplot as plt
import numpy as np

def plot_referral_curve(results):
    plt.style.use("ggplot")
    
    # Creiamo una figura con 2 grafici affiancati
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.array(results.referral_rates) * 100
    
    # --- PLOT 1: Accuratezza & AUSE ---
    ax1.plot(x, results.model_accuracies, marker='o', markersize=3, linestyle='-', 
             linewidth=2, color='tab:blue', label='Model Accuracy (Retained)')
    ax1.plot(x, results.oracle_accuracies, linestyle='--', linewidth=2, 
             color='tab:green', label='Oracle (Ideal)')
    ax1.plot(x, results.system_accuracies, linestyle='-.', linewidth=2,
             color='tab:purple', label='System Accuracy (AI + Doctor)')
             
    ax1.fill_between(x, results.model_accuracies, results.oracle_accuracies, 
                     color='gray', alpha=0.2, label=f'AUSE: {results.ause:.4f}')

    ax1.set_title(f"Accuracy vs Referral Rate\n({results.unc_method})")
    ax1.set_xlabel("Referral Rate (%)")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower left')
    ax1.grid(True)
    
    # --- PLOT 2: Cost Analysis ---
    costs = np.array(results.total_costs)
    min_cost_idx = np.argmin(costs)
    min_cost_val = costs[min_cost_idx]
    opt_rate = x[min_cost_idx]
    
    ax2.plot(x, costs, marker='s', markersize=3, color='tab:red', linewidth=2, label='Total Cost')
    
    # Evidenzia il minimo
    ax2.plot(opt_rate, min_cost_val, marker='*', markersize=15, color='gold', 
             markeredgecolor='black', label=f'Optimal: {opt_rate:.1f}%')
    
    d_acc = results.params.get('doctor_acc', 0)
    c_err = results.params.get('cost_err', 0)
    
    ax2.set_title(f"Cost Analysis Simulation\nDoc Acc: {d_acc:.0%} | Err Cost: {c_err}")
    ax2.set_xlabel("Referral Rate (%)")
    ax2.set_ylabel("Total Cost (Virtual Currency)")
    ax2.set_xlim(0, 100)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    filename = f"Referral_Analysis_{results.title.replace(' ', '_')}_{results.unc_method}.jpg"
    plt.savefig(filename, dpi=150)
    print(f"Grafico salvato: {filename}")
    plt.close()

def plot_analysis_dashboard(results):
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    x = np.array(results.referral_rates) * 100
    
    # --- GRAFICO 3: PERFORMANCE COMPARISON (Standalone vs Hybrid) ---
    baseline_ai = results.model_accuracies[0]
    doctor_acc = results.params.get('doctor_acc', 1.0)
    
    ax1.axhline(y=baseline_ai, color='tab:blue', linestyle='--', alpha=0.6, label=f'AI Standalone ({baseline_ai:.1%})')
    ax1.axhline(y=doctor_acc, color='tab:red', linestyle='--', alpha=0.6, label=f'Doctor Standalone ({doctor_acc:.1%})')
    ax1.plot(x, results.system_accuracies, color='tab:purple', linewidth=3, label='Hybrid System (AI + Doc)')
    
    ax1.set_title("Standalone vs Hybrid Comparison", fontsize=12)
    ax1.set_xlabel("Referral Rate (%)")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # --- GRAFICO 4: ERROR REJECTION EFFICIENCY ---
    # Calcoliamo quanti errori totali aveva l'IA all'inizio
    total_errors_initial = 1.0 - results.model_accuracies[0]
    
    # Calcoliamo quanti errori sono rimasti nel set IA ad ogni step
    # Errori rimanenti = (1 - Accuracy_retained) * (Percentuale dati rimasti)
    errors_remaining = [(1.0 - acc) * (1.0 - rate) for acc, rate in zip(results.model_accuracies, results.referral_rates)]
    
    # Errori catturati = Errori iniziali - Errori rimanenti
    errors_caught = [total_errors_initial - err for err in errors_remaining]
    
    # Percentuale di errori catturati rispetto al totale
    rejection_efficiency = [(e / total_errors_initial) * 100 if total_errors_initial > 0 else 0 for e in errors_caught]

    ax2.plot(x, rejection_efficiency, color='tab:orange', linewidth=3, label='Errors Referred to Doctor')
    ax2.plot(x, x, color='gray', linestyle=':', alpha=0.5, label='Random Rejection (Baseline)') # Diagonale casuale
    
    # Area tra la curva e la diagonale (mostra il guadagno rispetto al caso casuale)
    ax2.fill_between(x, rejection_efficiency, x, color='tab:orange', alpha=0.1)

    ax2.set_title("Error Rejection Efficiency", fontsize=12)
    ax2.set_xlabel("Referral Rate (% di dati inviati)")
    ax2.set_ylabel("% di Errori IA catturati")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"Referral_Analysis_Dashboard_{results.unc_method}.jpg"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Dashboard di analisi salvata: {filename}")