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
