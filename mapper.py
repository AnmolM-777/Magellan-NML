from simulator import MQCAModel, CMOSModel

class WorkloadMapper:
    def __init__(self, array_size=1024, mem_energy_pJ_per_byte=20.0):
        self.mqca = MQCAModel()
        self.cmos = CMOSModel()
        self.array_size = array_size
        self.mem_energy_J_per_byte = mem_energy_pJ_per_byte * 1e-12

    def map_workload(self, profiled_layers, bit_width=8):
        results = []
        for l in profiled_layers:
            # Basic calculation
            nml_energy = l['macs'] * 1e-18
            results.append({
                'layer_id': l['layer_id'],
                'nml_total_energy_J': nml_energy
            })
        return results
