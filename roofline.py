"""
roofline.py: Roofline Model and Bottleneck Analysis.
Computes compute ceilings, bandwidth bounds (LPDDR5, DDR5, HBM3), and plots curves.
Supports sensitivity analysis for NML frequency improvements.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

class RooflineAnalyzer:
    def __init__(self, memory_bandwidth_GBs=51.2):
        self.bandwidth_Bs = memory_bandwidth_GBs * 1e9
        # Multiple memory interfaces for comparative study
        self.mem_interfaces = {
            "LPDDR5": 42.6e9,
            "DDR5": 51.2e9,
            "HBM3": 819.2e9
        }

    def classify_layer_bottleneck(self, arithmetic_intensity, performance_ceiling_MACs):
        ridge_point = performance_ceiling_MACs / self.bandwidth_Bs
        if arithmetic_intensity < ridge_point:
            return "Memory-Bound"
        else:
            return "Compute-Bound"

    def generate_roofline_plots(self, mapped_results, nml_array_size, baseline_freq_hz, target_freq_hz, output_dir="plots"):
        os.makedirs(output_dir, exist_ok=True)
        
        # ----------------------------------------------------
        # Plot 1: Comparative Roofline Analysis with Multi-Memory
        # ----------------------------------------------------
        plt.figure(figsize=(11, 7))
        
        intensities = np.logspace(-2, 3, 500)
        
        # NML Baseline Roofline (100 MHz)
        nml_base_ceiling_GMACs = (nml_array_size * baseline_freq_hz) / 1e9
        nml_base_roofline = np.minimum(nml_base_ceiling_GMACs, intensities * (self.bandwidth_Bs / 1e9))
        plt.loglog(intensities, nml_base_roofline, 'b-', label=f'NML Baseline (100 MHz) - Max {nml_base_ceiling_GMACs:.1f} GMACs/s')
        
        # NML 2x Sensitivity Roofline (200 MHz)
        nml_target_ceiling_GMACs = (nml_array_size * target_freq_hz) / 1e9
        nml_target_roofline = np.minimum(nml_target_ceiling_GMACs, intensities * (self.bandwidth_Bs / 1e9))
        plt.loglog(intensities, nml_target_roofline, 'c--', label=f'NML 2x Sensitivity (200 MHz) - Max {nml_target_ceiling_GMACs:.1f} GMACs/s')

        # Draw the three multi-memory bandwidth diagonals
        colors = {"LPDDR5": "g", "DDR5": "r", "HBM3": "m"}
        styles = {"LPDDR5": ":", "DDR5": "-.", "HBM3": "-"}
        for name, bw in self.mem_interfaces.items():
            plt.loglog(intensities, intensities * (bw / 1e9), color=colors[name], linestyle=styles[name], 
                        alpha=0.6, label=f'{name} Bandwidth ({bw/1e9:.1f} GB/s)')

        # Plot individual layers
        plotted_models = set()
        model_colors = {'MiDaS': 'darkorange', 'YOLOv8-nano': 'forestgreen', 'MobileNetV2-Proxy': 'purple'}
        model_markers = {'MiDaS': 'o', 'YOLOv8-nano': 's', 'MobileNetV2-Proxy': '^'}

        for layer in mapped_results:
            model = layer["model"]
            ai = layer["arithmetic_intensity"]
            perf_gmacs = (layer["macs"] / layer["nml_time_sec"]) / 1e9
            
            label = f"{model} Layers" if model not in plotted_models else ""
            plotted_models.add(model)
            
            m_color = model_colors.get(model, 'blue')
            m_marker = model_markers.get(model, 'o')
            
            plt.scatter(ai, perf_gmacs, color=m_color, marker=m_marker, 
                        edgecolors='k', s=80, alpha=0.8, label=label, zorder=5)

        plt.title("Multi-Memory Roofline Model: Robotic Workloads on Nanomagnetic Logic", fontsize=14, fontweight='bold')
        plt.xlabel("Arithmetic Intensity (MACs / Byte)", fontsize=12)
        plt.ylabel("Attainable Performance (GMACs / s)", fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend(loc="lower right", fontsize=9)
        plt.xlim(1e-2, 1e2)
        plt.ylim(1e-1, 1e3)
        plt.tight_layout()
        
        roofline_path = os.path.join(output_dir, "roofline_model.png")
        plt.savefig(roofline_path, dpi=300)
        plt.close()
        
        # ----------------------------------------------------
        # Plot 2: Total Upgraded Energy Comparison
        # ----------------------------------------------------
        plt.figure(figsize=(13, 6))
        
        layer_ids = [l["layer_id"].split(".")[-1] + f" ({l['model'][:4]})" for l in mapped_results]
        nml_energies = [l["nml_total_energy_J"] * 1e6 for l in mapped_results]
        cmos_energies = [l["cmos_total_energy_J"] * 1e6 for l in mapped_results]
        
        x = np.arange(len(layer_ids))
        width = 0.35
        
        plt.bar(x - width/2, cmos_energies, width, label='28nm CMOS (Total)', color='lightcoral', edgecolor='k')
        plt.bar(x + width/2, nml_energies, width, label='MQCA/NML (Total)', color='skyblue', edgecolor='k')
        
        plt.title("Upgraded Energy Comparison per Layer: CMOS vs. Nanomagnetic (MQCA)", fontsize=14, fontweight='bold')
        plt.xlabel("Workload Layers", fontsize=12)
        plt.ylabel("Total Energy (microjoules - uJ)", fontsize=12)
        plt.xticks(x, layer_ids, rotation=45, ha='right', fontsize=8)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        energy_path = os.path.join(output_dir, "energy_comparison.png")
        plt.savefig(energy_path, dpi=300)
        plt.close()

        print(f"Generated and saved upgraded plots to: {output_dir}/")
        return roofline_path, energy_path

    def run_sensitivity_analysis(self, mapped_results, nml_array_size, baseline_freq, target_freq):
        table = []
        for l in mapped_results:
            ai = l["arithmetic_intensity"]
            base_bottleneck = self.classify_layer_bottleneck(ai, nml_array_size * baseline_freq)
            target_bottleneck = self.classify_layer_bottleneck(ai, nml_array_size * target_freq)
            
            status = "No Change"
            if base_bottleneck != target_bottleneck:
                status = f"Flipped ({base_bottleneck} -> {target_bottleneck})"
                
            table.append({
                "layer_id": l["layer_id"],
                "model": l["model"],
                "arithmetic_intensity": ai,
                "base_bottleneck": base_bottleneck,
                "target_bottleneck": target_bottleneck,
                "status": status
            })
        return table
