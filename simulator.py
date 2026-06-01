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
