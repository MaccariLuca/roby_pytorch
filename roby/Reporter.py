import os
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class HTMLReporter:
    def __init__(self, filename="roby_report.html"):
        self.filename = filename
        self.css = """
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; color: #333; }
            .container { max-width: 1200px; margin: auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
            
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 30px; }
            h2 { color: #34495e; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 15px; }
            
            /* Metriche Grid */
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .metric-box { background: #f1f8ff; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #d1e3f8; }
            .metric-value { font-size: 28px; font-weight: bold; color: #2980b9; }
            .metric-label { font-size: 14px; color: #57606f; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }
            .metric-sub { font-size: 11px; color: #95a5a6; margin-top: 5px; font-style: italic; }

            /* Tabelle */
            table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }
            th, td { padding: 12px 15px; border-bottom: 1px solid #eee; text-align: right; }
            th { background-color: #3498db; color: white; text-align: right; font-weight: 600; }
            th:first-child, td:first-child { text-align: left; }
            tr:nth-child(even) { background-color: #fcfcfc; }
            .optimal-row { background-color: #fff8e1 !important; font-weight: bold; border-left: 4px solid #f1c40f; }

            /* Immagini */
            .chart-container { text-align: center; margin: 30px 0; padding: 10px; background: white; border-radius: 8px; }
            img { max-width: 100%; height: auto; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
            
            .footer { margin-top: 60px; font-size: 12px; color: #bdc3c7; text-align: center; border-top: 1px solid #eee; padding-top: 20px; }
        </style>
        """

    def _fig_to_base64(self, fig):
        """Converte una figura Matplotlib in stringa base64 per l'HTML."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    def _plot_robustness_charts(self, results):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        metrics = [results.accuracies, results.thresholded_accuracies, 
                   results.weighted_accuracies, results.ncas]
        titles = ["Traditional Acc", "Thresholded Acc", "Weighted Acc", "Net Certainty"]
        
        steps = results.steps
        for i, ax in enumerate(axes):
            ax.plot(steps, metrics[i], marker='o', markersize=4, color='#3498db')
            ax.set_title(titles[i], fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 3: ax.set_ylim(-1, 1.05)
            else: ax.set_ylim(0, 1.05)
            
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_referral_charts(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        x = np.array(results.referral_rates) * 100
        min_idx = np.argmin(results.total_costs)
        
        # Plot 1: Accuratezza + Stella
        ax1.plot(x, results.model_accuracies, label='AI Model (Retained)', color='#3498db', linewidth=2)
        ax1.plot(x, results.system_accuracies, label='System (AI + Doctor)', color='#8e44ad', linestyle='-.', linewidth=2)
        ax1.plot(x, results.oracle_accuracies, label='Oracle', color='#2ecc71', linestyle='--', alpha=0.7)
        # STELLA SU ACCURATEZZA SISTEMA
        ax1.plot(x[min_idx], results.system_accuracies[min_idx], marker='*', markersize=15, 
                 color='gold', markeredgecolor='black', zorder=5, label='Optimal Point')
        
        ax1.set_title("Accuracy Analysis")
        ax1.set_xlabel("Referral Rate (%)")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Costi + Stella
        costs = np.array(results.total_costs)
        ax2.plot(x, costs, color='#e74c3c', linewidth=2, label='Total Cost')
        ax2.plot(x[min_idx], costs[min_idx], marker='*', markersize=15, color='gold', 
                 markeredgecolor='black', zorder=5, label='Optimal Cost')
        
        ax2.set_title("Cost Analysis Simulation")
        ax2.set_xlabel("Referral Rate (%)")
        ax2.set_ylabel("Total Cost")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_time_efficiency_section(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        x = np.array(results.referral_rates) * 100
        min_idx = np.argmin(results.total_costs)
        
        # --- GRAFICO 1: TEMPI COMPARATIVI ---
        # Ibrido
        ax1.plot(x, results.referral_times, label='Hybrid (Weighted)', color='#f39c12', linewidth=3)
        ax1.plot(x, results.extra_metrics.get('referral_times_fixed', []), label='Hybrid (Fixed)', color='#f39c12', linestyle='--', alpha=0.6)
        # Umano
        ax1.axhline(y=results.baseline_times['doc'], color='#e74c3c', label='Human (Weighted)', linewidth=2)
        ax1.axhline(y=results.baseline_times.get('doc_fixed', 0), color='#e74c3c', label='Human (Fixed)', linestyle='--', alpha=0.6)
        # Stella
        ax1.plot(x[min_idx], results.referral_times[min_idx], marker='*', markersize=15, color='gold', markeredgecolor='black', zorder=5)
        
        ax1.set_title("Average Time per Case (Weighted vs Fixed)")
        ax1.set_ylabel("Seconds")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # --- GRAFICO 2: RISPARMIO % ---
        doc_time = results.baseline_times['doc']
        efficiency_gain = [(doc_time - t) / doc_time * 100 for t in results.referral_times]
        ax2.bar(x, efficiency_gain, width=x[1]-x[0] if len(x)>1 else 1.0, color='#27ae60', alpha=0.6, label='Time Saved')
        ax2.set_title("Time Efficiency Gain (%)")
        ax2.set_ylabel("% Reduction vs Human Weighted")
        ax2.set_ylim(0, 105); ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_analysis_dashboard_base64(self, results):
        plt.style.use("ggplot")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        x = np.array(results.referral_rates) * 100
        min_idx = np.argmin(results.total_costs)
        
        # GRAFICO 1: PERFORMANCE COMPARISON + Stella
        baseline_ai = results.model_accuracies[0]
        doctor_acc = getattr(results, 'params', {}).get('doctor_acc', 1.0)
        ax1.axhline(y=baseline_ai, color='#3498db', linestyle='--', alpha=0.6, label=f'AI Standalone ({baseline_ai:.1%})')
        ax1.axhline(y=doctor_acc, color='#e74c3c', linestyle='--', alpha=0.6, label=f'Doctor Standalone ({doctor_acc:.1%})')
        ax1.plot(x, results.system_accuracies, color='#8e44ad', linewidth=3, label='Hybrid System')
        ax1.plot(x[min_idx], results.system_accuracies[min_idx], marker='*', markersize=15, color='gold', markeredgecolor='black', zorder=5)
        
        ax1.set_title("Standalone vs Hybrid Comparison")
        ax1.set_xlabel("Referral Rate (%)")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(min(baseline_ai, doctor_acc) - 0.05, 1.02)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # GRAFICO 2: ERROR REJECTION EFFICIENCY + Stella
        total_errors_initial = 1.0 - baseline_ai
        if total_errors_initial > 0:
            errors_remaining = [(1.0 - acc) * (1.0 - rate) for acc, rate in zip(results.model_accuracies, results.referral_rates)]
            errors_caught = [total_errors_initial - err for err in errors_remaining]
            rejection_efficiency = [(e / total_errors_initial) * 100 for e in errors_caught]
        else:
            rejection_efficiency = [0] * len(x)

        ax2.plot(x, rejection_efficiency, color='#f39c12', linewidth=3, label='Errors Referred')
        ax2.plot(x, x, color='gray', linestyle=':', alpha=0.5, label='Random Rejection')
        ax2.plot(x[min_idx], rejection_efficiency[min_idx], marker='*', markersize=15, color='gold', markeredgecolor='black', zorder=5)
        ax2.fill_between(x, rejection_efficiency, x, color='#f39c12', alpha=0.1)

        ax2.set_title("Error Rejection Efficiency")
        ax2.set_xlabel("Referral Rate (%)")
        ax2.set_ylabel("% of AI Errors Intercepted")
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 105)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def generate_report(self, results, mode="Robustness"):
        """ Genera il report HTML completo. """
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        html_head = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Roby HTML Report - {mode}</title>{self.css}</head>
        <body><div class="container"><div style="display:flex; justify-content:space-between; align-items:center;">
        <h1>Analysis: {mode}</h1><div style="text-align:right; color:#7f8c8d;"><div>{timestamp}</div></div></div>"""

        html_body = ""
        if mode == "Robustness":
            html_body += f'<div class="metrics-grid"><div class="metric-box"><div class="metric-value">{np.mean(results.accuracies):.2%}</div><div class="metric-label">Avg Accuracy</div></div><div class="metric-box"><div class="metric-value">{results.robustness:.2%}</div><div class="metric-label">Robustness Score</div></div></div>'
            fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(results.steps, results.accuracies, marker='o', color='#3498db'); chart_b64 = self._fig_to_base64(fig)
            html_body += f'<h2>Visualization</h2><div class="chart-container"><img src="data:image/png;base64,{chart_b64}"></div>'

        elif mode == "Uncertainty":
            html_body += f'<div class="metrics-grid"><div class="metric-box"><div class="metric-value">{np.mean(results.accuracies):.2%}</div><div class="metric-label">Avg Accuracy</div></div><div class="metric-box"><div class="metric-value">{np.mean(results.eces):.3f}</div><div class="metric-label">Avg ECE</div></div></div>'
            chart_b64 = self._plot_robustness_charts(results)
            html_body += f'<h2>Visualization</h2><div class="chart-container"><img src="data:image/png;base64,{chart_b64}"></div>'

        elif mode == "Referral":
            params = getattr(results, 'params', {})
            extra = getattr(results, 'extra_metrics', {})
            min_cost_idx = np.argmin(results.total_costs)
            
            # --- KPI PRINCIPALI ---
            html_body += f"""
                <h2>Key Performance Indicators (at Optimal Point)</h2>
                <div class="metrics-grid">
                    <div class="metric-box" style="border-color:#f1c40f; background:#fffcf0;"><div class="metric-value" style="color:#f39c12;">{results.referral_rates[min_cost_idx]:.1%}</div><div class="metric-label">Optimal Referral Rate</div></div>
                    <div class="metric-box"><div class="metric-value">{min(results.total_costs):.1f}</div><div class="metric-label">Minimum Cost</div></div>
                    <div class="metric-box"><div class="metric-value">{results.system_accuracies[min_cost_idx]:.2%}</div><div class="metric-label">System Accuracy</div></div>
                    <div class="metric-box"><div class="metric-value">{results.ause:.4f}</div><div class="metric-label">AUSE Score</div><div class="metric-sub">Lower is better</div></div>
                </div>
                <h2>Error Rejection & Optimization</h2>
                <div class="metrics-grid">
                    <div class="metric-box"><div class="metric-value">{extra.get('err_ref_pct', 0):.1%}</div><div class="metric-label">AI Errors Referred</div><div class="metric-sub">Error Capture Rate</div></div>
                    <div class="metric-box"><div class="metric-value">{extra.get('fn_ref_pct', 0):.1%}</div><div class="metric-label">False Negatives Saved</div><div class="metric-sub">Critical Safety</div></div>
                    <div class="metric-box"><div class="metric-value">{extra.get('fp_ref_pct', 0):.1%}</div><div class="metric-label">False Positives Saved</div><div class="metric-sub">Efficiency on FPs</div></div>
                    <div class="metric-box" style="border-color:#2ecc71; background:#f0fff4;"><div class="metric-value" style="color:#27ae60;">{extra.get('correct_retained_pct', 0):.1%}</div><div class="metric-label">AI Corrects Retained</div><div class="metric-sub">Avoided Over-referral</div></div>
                </div>
            """
            
            # --- TABELLA ---
            html_body += """<h2>Detailed Simulation Data</h2><table><thead><tr><th>Referral Rate</th><th>AI Acc (Retained)</th><th>System Acc (Hybrid)</th><th>Total Cost</th><th>Threshold</th></tr></thead><tbody>"""
            step_size = max(1, len(results.referral_rates) // 20)
            indices = sorted(list(set(list(range(0, len(results.referral_rates), step_size)) + [min_cost_idx])))
            for i in indices:
                row_class = 'class="optimal-row"' if i == min_cost_idx else ""
                html_body += f'<tr {row_class}><td>{results.referral_rates[i]:.1%} {"⭐" if i == min_cost_idx else ""}</td><td>{results.model_accuracies[i]:.4f}</td><td>{results.system_accuracies[i]:.4f}</td><td>{results.total_costs[i]:.1f}</td><td>{results.thresholds[i]:.4f}</td></tr>'
            html_body += "</tbody></table>"

            # --- GRAFICI ---
            chart_b64 = self._plot_referral_charts(results)
            time_b64 = self._plot_time_efficiency_section(results)
            dash_b64 = self._plot_analysis_dashboard_base64(results)
            html_body += f'<h2>Visualizations</h2><div class="chart-container"><h3>Cost & Accuracy Analysis</h3><img src="data:image/png;base64,{chart_b64}"></div>'
            html_body += f'<div class="chart-container"><h3>Time Efficiency Analysis</h3><img src="data:image/png;base64,{time_b64}"></div>'
            html_body += f'<div class="chart-container"><h3>Hybrid System Comparison</h3><img src="data:image/png;base64,{dash_b64}"></div>'

        html_content = html_head + html_body + '<div class="footer">Generated by Roby Framework | Referral Analysis Report</div></div></body></html>'
        with open(self.filename, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"Report HTML generato: {os.path.abspath(self.filename)}")