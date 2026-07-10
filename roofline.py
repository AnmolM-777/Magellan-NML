"""
roofline.py: Roofline Model and Bottleneck Analysis.
Computes compute ceilings, bandwidth bounds, and plots Roofline curves and energy comparisons.
Supports sensitivity analysis for NML frequency improvements.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent GUI errors
import matplotlib.pyplot as plt

class RooflineAnalyzer:
    def __init__(self, memory_bandwidth_GBs=51.2):
        """
        :param memory_bandwidth_GBs: DRAM memory bandwidth in GB/s (default 51.2 GB/s for DDR5).
        """
        self.bandwidth_Bs = memory_bandwidth_GBs * 1e9

    def classify_layer_bottleneck(self, arithmetic_intensity, performance_ceiling_MACs):
        """
        Classifies if a layer is memory-bound or compute-bound under given parameters.
        """
        ridge_point = performance_ceiling_MACs / self.bandwidth_Bs  # MACs/Byte
        if arithmetic_intensity < ridge_point:
            return "Memory-Bound"
        else:
            return "Compute-Bound"

    def generate_roofline_plots(self, mapped_results, nml_array_size, baseline_freq_hz, target_freq_hz, output_dir="plots"):
        """
        Generates and saves:
        1. A comparative Roofline plot (Baseline 100MHz vs 2x Sensitivity 200MHz vs CMOS 1GHz).
        2. An energy comparison bar chart (NML vs CMOS).
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # ----------------------------------------------------
        # Plot 1: Roofline Analysis
        # ----------------------------------------------------
        plt.figure(figsize=(10, 6))
        
        # Define intensities range for plotting curves
        intensities = np.logspace(-2, 3, 500)  # MACs/Byte
        
        # Platform 1: NML Baseline (e.g., 100 MHz)
        nml_base_ceiling_MACs = nml_array_size * baseline_freq_hz
        nml_base_ceiling_GMACs = nml_base_ceiling_MACs / 1e9
        nml_base_roofline = np.minimum(nml_base_ceiling_GMACs, intensities * (self.bandwidth_Bs / 1e9))
        plt.loglog(intensities, nml_base_roofline, 'b-', label=f'NML Baseline ({baseline_freq_hz/1e6:.0f} MHz) - Max {nml_base_ceiling_GMACs:.1f} GMACs/s')
        
        # Platform 2: NML 2x Speed (e.g., 200 MHz)
        nml_target_ceiling_MACs = nml_array_size * target_freq_hz
        nml_target_ceiling_GMACs = nml_target_ceiling_MACs / 1e9
        nml_target_roofline = np.minimum(nml_target_ceiling_GMACs, intensities * (self.bandwidth_Bs / 1e9))
        plt.loglog(intensities, nml_target_roofline, 'c--', label=f'NML 2x Sensitivity ({target_freq_hz/1e6:.0f} MHz) - Max {nml_target_ceiling_GMACs:.1f} GMACs/s')

        # Memory bandwidth diagonal line
        plt.loglog(intensities, intensities * (self.bandwidth_Bs / 1e9), 'r:', alpha=0.5, label=f'Memory Bandwidth ({self.bandwidth_Bs/1e9:.1f} GB/s)')

        # Plot individual layers as scatter points
        plotted_models = set()
        colors = {'MiDaS': 'darkorange', 'YOLOv8-nano': 'forestgreen'}
        markers = {'MiDaS': 'o', 'YOLOv8-nano': 's'}

        for layer in mapped_results:
            model = layer["model"]
            ai = layer["arithmetic_intensity"]
            # Actual performance in GMACs/sec = MACs / (time_sec * 1e9)
            perf_gmacs = (layer["macs"] / layer["nml_time_sec"]) / 1e9
            
            label = f"{model} Layers" if model not in plotted_models else ""
            plotted_models.add(model)
            
            plt.scatter(ai, perf_gmacs, color=colors[model], marker=markers[model], 
                        edgecolors='k', s=70, alpha=0.8, label=label, zorder=5)

        # Plot labels & styling
        plt.title("Roofline Model: Attainable Performance of Robotic Workloads on NML", fontsize=14, fontweight='bold')
        plt.xlabel("Arithmetic Intensity (MACs / Byte)", fontsize=12)
        plt.ylabel("Attainable Performance (GMACs / s)", fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=10)
        plt.xlim(1e-2, 1e2)
        plt.ylim(1e-1, 1e3)
        plt.tight_layout()
        
        roofline_path = os.path.join(output_dir, "roofline_model.png")
        plt.savefig(roofline_path, dpi=300)
        plt.close()
        
        # ----------------------------------------------------
        # Plot 2: Energy Comparison
        # ----------------------------------------------------
        plt.figure(figsize=(12, 6))
        
        layer_ids = [l["layer_id"].split(".")[-1] + f" ({l['model'][:4]})" for l in mapped_results]
        nml_energies = [l["nml_total_energy_J"] * 1e6 for l in mapped_results]  # microjoules (uJ)
        cmos_energies = [l["cmos_total_energy_J"] * 1e6 for l in mapped_results]  # microjoules (uJ)
        
        x = np.arange(len(layer_ids))
        width = 0.35
        
        plt.bar(x - width/2, cmos_energies, width, label='28nm CMOS (Total)', color='lightcoral', edgecolor='k')
        plt.bar(x + width/2, nml_energies, width, label='MQCA/NML (Total)', color='skyblue', edgecolor='k')
        
        plt.title("Total Energy Comparison per Layer: CMOS vs. Nanomagnetic (MQCA)", fontsize=14, fontweight='bold')
        plt.xlabel("Workload Layers", fontsize=12)
        plt.ylabel("Total Energy (microjoules - uJ)", fontsize=12)
        plt.xticks(x, layer_ids, rotation=45, ha='right', fontsize=9)
        plt.yscale('log')  # Log scale since energy spans orders of magnitude
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        energy_path = os.path.join(output_dir, "energy_comparison.png")
        plt.savefig(energy_path, dpi=300)
        plt.close()

        print(f"Generated and saved analysis plots to: {output_dir}/")
        return roofline_path, energy_path

    def run_sensitivity_analysis(self, mapped_results, nml_array_size, baseline_freq, target_freq):
        """
        Tabulates which layers shift bottleneck state when clock speed changes.
        """
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
