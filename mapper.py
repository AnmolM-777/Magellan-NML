"""
mapper.py: Workload Mapping and Energy/Performance Estimation.
Maps profiled layers onto MQCA and CMOS hardware resources, calculating
compute energy, memory access energy, and total execution latency.
"""

from simulator import MQCAModel, CMOSModel

class WorkloadMapper:
    def __init__(self, nml_array_size=1024, nml_freq_hz=100e6, 
                 cmos_array_size=1024, cmos_freq_hz=1e9,
                 mem_energy_pJ_per_byte=20.0):
        """
        :param nml_array_size: Number of parallel MAC units in the MQCA array.
        :param nml_freq_hz: Clock frequency of the MQCA array (default 100 MHz).
        :param cmos_array_size: Number of parallel MAC units in the CMOS array.
        :param cmos_freq_hz: Clock frequency of the CMOS baseline (default 1 GHz).
        :param mem_energy_pJ_per_byte: Energy required to read/write 1 Byte from/to DRAM (DDR5 benchmark is ~20 pJ/B).
        """
        self.nml_array_size = nml_array_size
        self.nml_freq_hz = nml_freq_hz
        self.cmos_array_size = cmos_array_size
        self.cmos_freq_hz = cmos_freq_hz
        self.mem_energy_J_per_byte = mem_energy_pJ_per_byte * 1e-12

        self.mqca_model = MQCAModel()
        self.cmos_model = CMOSModel()

    def map_workload(self, profiled_layers, bit_width=8):
        """
        Maps each layer in the profile and calculates performance metrics.
        """
        mapped_results = []
        mqca_mac_specs = self.mqca_model.get_mac_specs(bit_width)
        
        for layer in profiled_layers:
            macs = layer["macs"]
            mem_bytes = layer["total_memory_bytes"]

            # ----------------------------------------------------
            # 1. MQCA (NML) Calculations
            # ----------------------------------------------------
            # Compute cycles = ceil(macs / array_size) + pipeline depth
            compute_steps = (macs + self.nml_array_size - 1) // self.nml_array_size
            nml_total_cycles = compute_steps + mqca_mac_specs["latency_cycles"]
            nml_compute_time_sec = nml_total_cycles / self.nml_freq_hz
            
            # Energy calculations
            nml_compute_energy_J = macs * mqca_mac_specs["energy_J"]
            mem_energy_J = mem_bytes * self.mem_energy_J_per_byte
            nml_total_energy_J = nml_compute_energy_J + mem_energy_J

            # ----------------------------------------------------
            # 2. CMOS Baseline Calculations
            # ----------------------------------------------------
            cmos_compute_steps = (macs + self.cmos_array_size - 1) // self.cmos_array_size
            cmos_mac_latency_cycles = 5 # Standard CMOS pipeline stages
            cmos_total_cycles = cmos_compute_steps + cmos_mac_latency_cycles
            cmos_compute_time_sec = cmos_total_cycles / self.cmos_freq_hz
            
            cmos_compute_energy_J = macs * self.cmos_model.get_mac_energy_J(bit_width)
            cmos_total_energy_J = cmos_compute_energy_J + mem_energy_J

            # ----------------------------------------------------
            # 3. Comparative Metrics
            # ----------------------------------------------------
            # Compute-only energy savings vs Total energy savings (including memory)
            compute_energy_savings_ratio = cmos_compute_energy_J / nml_compute_energy_J if nml_compute_energy_J > 0 else 1
            total_energy_savings_ratio = cmos_total_energy_J / nml_total_energy_J if nml_total_energy_J > 0 else 1
            
            # Latency overhead (NML is typically slower due to frequency)
            latency_slowdown_ratio = nml_compute_time_sec / cmos_compute_time_sec if cmos_compute_time_sec > 0 else 1

            mapped_results.append({
                "layer_id": layer["layer_id"],
                "model": layer["model"],
                "type": layer["type"],
                "macs": macs,
                "total_memory_bytes": mem_bytes,
                "arithmetic_intensity": layer["arithmetic_intensity"],
                
                # MQCA detail
                "nml_latency_cycles": nml_total_cycles,
                "nml_time_sec": nml_compute_time_sec,
                "nml_compute_energy_J": nml_compute_energy_J,
                "nml_total_energy_J": nml_total_energy_J,
                
                # CMOS detail
                "cmos_latency_cycles": cmos_total_cycles,
                "cmos_time_sec": cmos_compute_time_sec,
                "cmos_compute_energy_J": cmos_compute_energy_J,
                "cmos_total_energy_J": cmos_total_energy_J,
                
                # Shared
                "mem_energy_J": mem_energy_J,
                
                # Ratios
                "compute_energy_savings": compute_energy_savings_ratio,
                "total_energy_savings": total_energy_savings_ratio,
                "latency_slowdown": latency_slowdown_ratio
            })

        return mapped_results

    def analyze_rankings(self, mapped_results):
        """
        Identifies top-3 layers for energy reduction and bottom-3 layers.
        """
        # Sort by total energy savings ratio (descending)
        sorted_by_savings = sorted(mapped_results, key=lambda x: x["total_energy_savings"], reverse=True)
        top_3 = sorted_by_savings[:3]
        bottom_3 = sorted_by_savings[-3:]
        return top_3, bottom_3

if __name__ == "__main__":
    from profiler import WorkloadProfiler
    prof = WorkloadProfiler()
    layers = prof.run_profiler(bit_width=8)
    
    mapper = WorkloadMapper()
    results = mapper.map_workload(layers, bit_width=8)
    top, bottom = mapper.analyze_rankings(results)
    
    print("\n--- TOP 3 LAYERS BY ENERGY SAVINGS POTENTIAL ---")
    for l in top:
        print(f"Layer: {l['layer_id']} ({l['type']}) -> Savings: {l['total_energy_savings']:.1f}x (Intensity: {l['arithmetic_intensity']:.2f})")
        
    print("\n--- BOTTOM 3 LAYERS BY ENERGY SAVINGS POTENTIAL ---")
    for l in bottom:
        print(f"Layer: {l['layer_id']} ({l['type']}) -> Savings: {l['total_energy_savings']:.1f}x (Intensity: {l['arithmetic_intensity']:.2f})")
