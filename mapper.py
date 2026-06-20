from simulator import MQCAModel, CMOSModel

class WorkloadMapper:
    def __init__(self, array_size=1024, mem_energy_pJ_per_byte=20.0):
        self.mqca_model = MQCAModel()
        self.cmos_model = CMOSModel()
        self.array_size = array_size
        self.mem_energy_J_per_byte = mem_energy_pJ_per_byte * 1e-12

    def map_workload(self, profiled_layers, bit_width=8):
        results = []
        mqca_mac_specs = self.mqca_model.get_mac_specs(bit_width)
        for l in profiled_layers:
            macs = l['macs']
            mem = l['total_memory_bytes']
            nml_compute = macs * mqca_mac_specs['energy_J']
            mem_energy = mem * self.mem_energy_J_per_byte
            nml_total = nml_compute + mem_energy
            cmos_compute = macs * self.cmos_model.get_mac_energy_J(bit_width)
            cmos_total = cmos_compute + mem_energy
            results.append({
                'layer_id': l['layer_id'],
                'model': l['model'],
                'macs': macs,
                'total_memory_bytes': mem,
                'arithmetic_intensity': l['arithmetic_intensity'],
                'nml_total_energy_J': nml_total,
                'cmos_total_energy_J': cmos_total,
                'total_energy_savings': cmos_total / nml_total if nml_total > 0 else 1.0
            })
        return results

    def analyze_rankings(self, mapped_results):
        sorted_by_savings = sorted(mapped_results, key=lambda x: x['total_energy_savings'], reverse=True)
        return sorted_by_savings[:3], sorted_by_savings[-3:]
