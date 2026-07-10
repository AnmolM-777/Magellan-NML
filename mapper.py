"""
mapper.py: Workload Mapping and Energy/Performance Estimation.
Upgraded with Cache Hierarchy (SRAM Scratchpad vs DRAM) and Heterogeneous Co-Processor Scheduling.
"""

from simulator import MQCAModel, CMOSModel

class WorkloadMapper:
    def __init__(self, nml_array_size=1024, nml_freq_hz=100e6, 
                 cmos_array_size=1024, cmos_freq_hz=1e9,
                 mem_energy_pJ_per_byte=20.0,
                 sram_cache_size_bytes=2 * 1024 * 1024,  # 2MB SRAM
                 sram_energy_pJ_per_byte=1.0):           # 1pJ/B SRAM energy
        """
        :param sram_cache_size_bytes: Size of on-chip SRAM buffer in Bytes.
        :param sram_energy_pJ_per_byte: Energy to read/write 1 Byte from local SRAM cache (default 1 pJ/B).
        """
        self.nml_array_size = nml_array_size
        self.nml_freq_hz = nml_freq_hz
        self.cmos_array_size = cmos_array_size
        self.cmos_freq_hz = cmos_freq_hz
        self.mem_energy_J_per_byte = mem_energy_pJ_per_byte * 1e-12
        self.sram_cache_size_bytes = sram_cache_size_bytes
        self.sram_energy_J_per_byte = sram_energy_pJ_per_byte * 1e-12

        self.mqca_model = MQCAModel()
        self.cmos_model = CMOSModel()

    def map_workload(self, profiled_layers, bit_width=8):
        """
        Maps each layer and calculates upgraded performance, cache metrics, and scheduling.
        """
        mapped_results = []
        mqca_mac_specs = self.mqca_model.get_mac_specs(bit_width)
        
        for layer in profiled_layers:
            macs = layer["macs"]
            mem_bytes = layer["total_memory_bytes"]

            # 1. Cache Hierarchy Evaluation
            # If total weights and activations fit in SRAM cache -> hit, else DRAM access
            if mem_bytes <= self.sram_cache_size_bytes:
                cache_status = "HIT (SRAM)"
                active_mem_energy_J_per_byte = self.sram_energy_J_per_byte
            else:
                cache_status = "MISS (DRAM)"
                active_mem_energy_J_per_byte = self.mem_energy_J_per_byte

            mem_energy_J = mem_bytes * active_mem_energy_J_per_byte

            # 2. MQCA Calculations
            compute_steps = (macs + self.nml_array_size - 1) // self.nml_array_size
            nml_total_cycles = compute_steps + mqca_mac_specs["latency_cycles"]
            nml_compute_time_sec = nml_total_cycles / self.nml_freq_hz
            
            nml_compute_energy_J = macs * mqca_mac_specs["energy_J"]
            nml_total_energy_J = nml_compute_energy_J + mem_energy_J

            # 3. CMOS Baseline Calculations
            cmos_compute_steps = (macs + self.cmos_array_size - 1) // self.cmos_array_size
            cmos_mac_latency_cycles = 5
            cmos_total_cycles = cmos_compute_steps + cmos_mac_latency_cycles
            cmos_compute_time_sec = cmos_total_cycles / self.cmos_freq_hz
            
            cmos_compute_energy_J = macs * self.cmos_model.get_mac_energy_J(bit_width)
            cmos_total_energy_J = cmos_compute_energy_J + mem_energy_J

            # 4. Comparative Metrics
            compute_energy_savings_ratio = cmos_compute_energy_J / nml_compute_energy_J if nml_compute_energy_J > 0 else 1
            total_energy_savings_ratio = cmos_total_energy_J / nml_total_energy_J if nml_total_energy_J > 0 else 1
            latency_slowdown_ratio = nml_compute_time_sec / cmos_compute_time_sec if cmos_compute_time_sec > 0 else 1

            # 5. Heterogeneous Scheduling Logic
            # Rule: Schedule to NML co-processor ONLY if it yields >= 4x compute energy savings
            # AND cache is a HIT (to protect from DRAM latency overheads). Otherwise run on CMOS.
            if (compute_energy_savings_ratio >= 4.0) and (cache_status == "HIT (SRAM)"):
                assigned_core = "MQCA_NML"
            else:
                assigned_core = "CMOS"

            mapped_results.append({
                "layer_id": layer["layer_id"],
                "model": layer["model"],
                "type": layer["type"],
                "macs": macs,
                "total_memory_bytes": mem_bytes,
                "arithmetic_intensity": layer["arithmetic_intensity"],
                
                # Cache stats
                "cache_status": cache_status,
                "mem_energy_J": mem_energy_J,
                
                # MQCA detail
                "nml_latency_cycles": nml_total_cycles,
                "nml_time_sec": nml_compute_time_sec,
                "nml_compute_energy_J": nml_compute_energy_J,
                "nml_total_energy_J": nml_total_energy_J,
                "nml_area_um2": mqca_mac_specs["area_um2"] * self.nml_array_size,
                "nml_error_rate": mqca_mac_specs["error_rate"],
                
                # CMOS detail
                "cmos_latency_cycles": cmos_total_cycles,
                "cmos_time_sec": cmos_compute_time_sec,
                "cmos_compute_energy_J": cmos_compute_energy_J,
                "cmos_total_energy_J": cmos_total_energy_J,
                "cmos_area_um2": self.cmos_model.get_mac_area_um2(bit_width) * self.cmos_array_size,
                
                # comparative & scheduling
                "compute_energy_savings": compute_energy_savings_ratio,
                "total_energy_savings": total_energy_savings_ratio,
                "latency_slowdown": latency_slowdown_ratio,
                "assigned_core": assigned_core
            })

        return mapped_results

    def analyze_rankings(self, mapped_results):
        sorted_by_savings = sorted(mapped_results, key=lambda x: x["total_energy_savings"], reverse=True)
        return sorted_by_savings[:3], sorted_by_savings[-3:]
