from typing import List

class UncertaintyResults:
    def __init__(self, steps: List[float], 
                 accuracies: List[float],
                 thresholded_accuracies: List[float],
                 weighted_accuracies: List[float],
                 au_urocs: List[float],
                 acuas: List[float],
                 ncas: List[float],
                 eces: List[float],  
                 title: str,
                 alteration_name: str,
                 times: List[float]=None):
        
        self.steps = steps
        self.accuracies = accuracies
        self.thresholded_accuracies = thresholded_accuracies
        self.weighted_accuracies = weighted_accuracies
        self.au_urocs = au_urocs
        self.acuas = acuas
        self.ncas = ncas
        self.eces = eces
        self.title = title
        self.alteration_name = alteration_name
        self.times = times

    def print_summary(self):
        print(f"\n--- Risultati Incertezza per: {self.title} ---")
        # Aggiunta colonna ECE
        print(f"{'Step':<6} {'Acc':<6} {'ThrAcc':<6} {'W-Acc':<6} {'AU-UROC':<8} {'ACUA':<6} {'NCA':<6} {'ECE':<6}")
        for i, s in enumerate(self.steps):
            print(f"{s:<6.2f} {self.accuracies[i]:<6.2f} {self.thresholded_accuracies[i]:<6.2f} "
                  f"{self.weighted_accuracies[i]:<6.2f} {self.au_urocs[i]:<8.2f} "
                  f"{self.acuas[i]:<6.2f} {self.ncas[i]:<6.2f} {self.eces[i]:<6.2f}") # <--- Stampa ECE