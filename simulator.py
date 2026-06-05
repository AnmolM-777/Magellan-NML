class MQCAModel:
    def __init__(self, energy_per_magnet_switch_aJ=2.0):
        self.energy_per_magnet_switch_aJ = energy_per_magnet_switch_aJ

    def get_adder_specs(self, bit_width):
        magnet_count = 35 * bit_width
        gate_count = 4 * bit_width
        latency_cycles = 0.5 * bit_width + 1.0
        return {
            'magnets': magnet_count,
            'gates': gate_count,
            'latency_cycles': latency_cycles
        }

    def get_multiplier_specs(self, bit_width):
        magnet_count = 30 * (bit_width ** 2)
        gate_count = 3 * (bit_width ** 2)
        latency_cycles = 1.5 * bit_width
        return {
            'magnets': magnet_count,
            'gates': gate_count,
            'latency_cycles': latency_cycles
        }

    def get_mac_specs(self, bit_width):
        mult_specs = self.get_multiplier_specs(bit_width)
        acc_bit_width = 2 * bit_width + 4
        add_specs = self.get_adder_specs(acc_bit_width)
        total_magnets = mult_specs['magnets'] + add_specs['magnets']
        energy_aJ = total_magnets * self.energy_per_magnet_switch_aJ
        return {
            'magnets': total_magnets,
            'energy_aJ': energy_aJ,
            'energy_J': energy_aJ * 1e-18
        }

class CMOSModel:
    def __init__(self, process_node_nm=28):
        self.process_node_nm = process_node_nm

    def get_mac_energy_J(self, bit_width):
        if bit_width <= 4:
            energy_fJ = 35.0
        elif bit_width <= 8:
            energy_fJ = 150.0
        else:
            energy_fJ = 400.0
        # BUG: Multiplying by 1e-12 instead of 1e-15 (pJ instead of fJ)
        return energy_fJ * 1e-12
