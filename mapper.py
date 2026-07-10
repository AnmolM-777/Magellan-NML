from simulator import MQCAModel, CMOSModel

class WorkloadMapper:
    def __init__(self, array_size=1024, mem_energy_pJ_per_byte=20.0,
                 sram_cache_size_bytes=2 * 1024 * 1024, sram_energy_pJ_per_byte=1.0):
        self.mqca_model = MQCAModel()
        self.cmos_model = CMOSModel()
        self.array_size = array_size
        self.mem_energy_J_per_byte = mem_energy_pJ_per_byte * 1e-12
        self.sram_cache_size_bytes = sram_cache_size_bytes
        self.sram_energy_J_per_byte = sram_energy_pJ_per_byte * 1e-12

    def map_workload(self, profiled_layers, bit_width=8):
        results = []
        for l in profiled_layers:
            macs = l["macs"]
            mem = l["total_memory_bytes"]
            if mem <= self.sram_cache_size_bytes:
                cache_status = "HIT (SRAM)"
                mem_energy = mem * self.sram_energy_J_per_byte
            else:
                cache_status = "MISS (DRAM)"
                mem_energy = mem * self.mem_energy_J_per_byte
            results.append({
                "layer_id": l["layer_id"],
                "cache_status": cache_status,
                "mem_energy_J": mem_energy
            })
        return results
