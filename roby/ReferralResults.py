from typing import List

class ReferralResults:
    def __init__(self, 
                 referral_rates: List[float], 
                 model_accuracies: List[float], 
                 system_accuracies: List[float],  # Accuracy AI + Dottore
                 oracle_accuracies: List[float],
                 total_costs: List[float],        # Curva dei costi
                 ause: float,
                 thresholds: List[float],
                 title: str,
                 unc_method: str,
                 simulation_params: dict,
                 extra_metrics: dict = None,
                 referral_times: List[float] = None,
                 baseline_times: dict = None):      
        
        self.referral_rates = referral_rates
        self.model_accuracies = model_accuracies
        self.system_accuracies = system_accuracies
        self.oracle_accuracies = oracle_accuracies
        self.total_costs = total_costs
        self.ause = ause
        self.thresholds = thresholds
        self.title = title
        self.unc_method = unc_method
        self.params = simulation_params
        self.extra_metrics = extra_metrics or {}
        self.referral_times = referral_times or []
        self.baseline_times = baseline_times or {}

    def print_summary(self):
        print(f"\n--- Referral Analysis: {self.title} ---")
        print(f"Uncertainty Method: {self.unc_method}")
        print(f"Doctor Accuracy:    {self.params.get('doctor_acc', 'N/A'):.1%}")
        print(f"Cost Settings:      Err={self.params.get('cost_err')} | Doc={self.params.get('cost_doc')}")
        print(f"AUSE Score:         {self.ause:.5f} (Lower is better)")
        
        # Trova il punto di costo minimo
        min_cost_idx = self.total_costs.index(min(self.total_costs))
        optimal_rate = self.referral_rates[min_cost_idx]
        
        print("-" * 85)
        print(f"{'Ref. Rate':<10} {'Retained':<10} {'Model Acc':<12} {'System Acc':<12} {'Total Cost':<12} {'Threshold':<15}")
        
        steps_to_show = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, optimal_rate]
        # Ordina e rimuovi duplicati per la visualizzazione
        steps_to_show = sorted(list(set([round(s, 2) for s in steps_to_show])))

        for i, rate in enumerate(self.referral_rates):
            # Mostra i punti chiave e l'ottimo
            is_optimal = (i == min_cost_idx)
            marker = "*" if is_optimal else ""
            
            if round(rate, 2) in steps_to_show or i==0 or i==len(self.referral_rates)-1:
                retained = f"{(1-rate)*100:.0f}%"
                print(f"{rate:<10.2f} {retained:<10} {self.model_accuracies[i]:<12.4f} "
                      f"{self.system_accuracies[i]:<12.4f} {self.total_costs[i]:<12.1f}{marker} {self.thresholds[i]:<15.4f}")
        
        print("-" * 85)
        print(f"* Optimal Referral Rate based on Cost: {optimal_rate:.1%} (Cost: {min(self.total_costs):.1f})")