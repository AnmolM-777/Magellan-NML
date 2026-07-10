"""
simulator.py: Nanomagnetic Logic (NML) / MQCA & CMOS Baseline Simulation Engine.
Models arithmetic primitives and scales energy and latency metrics based on bit-width.
"""
import math

class MQCAModel:
    def __init__(self, energy_per_magnet_switch_aJ=2.0, switching_activity=0.5, temp_K=300.0):
        self.energy_per_magnet_switch_aJ = energy_per_magnet_switch_aJ
        self.switching_activity = switching_activity
        self.temp_K = temp_K
        self.k_B = 1.380649e-23
        # Landauer Limit Minimum: E = k_B * T * ln(2)
        self.thermal_min_energy_J = self.k_B * self.temp_K * math.log(2.0)
        self.thermal_min_energy_aJ = self.thermal_min_energy_J * 1e18

    def get_adder_specs(self, bit_width):
        magnet_count = 35 * bit_width
        gate_count = 4 * bit_width
        latency_cycles = 0.5 * bit_width + 1.0
        return {"magnets": magnet_count, "gates": gate_count, "latency_cycles": latency_cycles}

    def get_multiplier_specs(self, bit_width):
        magnet_count = 30 * (bit_width ** 2)
        gate_count = 3 * (bit_width ** 2)
        latency_cycles = 1.5 * bit_width
        return {"magnets": magnet_count, "gates": gate_count, "latency_cycles": latency_cycles}

    def get_mac_specs(self, bit_width):
        mult_specs = self.get_multiplier_specs(bit_width)
        acc_bit_width = 2 * bit_width + 4
        add_specs = self.get_adder_specs(acc_bit_width)
        total_magnets = mult_specs["magnets"] + add_specs["magnets"]
        energy_aJ = total_magnets * self.switching_activity * self.energy_per_magnet_switch_aJ
        
        # Enforce Landauer physics limit floor
        min_adiabatic_energy_aJ = total_magnets * self.switching_activity * self.thermal_min_energy_aJ
        if energy_aJ < min_adiabatic_energy_aJ:
            energy_aJ = min_adiabatic_energy_aJ
        
        return {
            "magnets": total_magnets,
            "gates": mult_specs["gates"] + add_specs["gates"],
            "latency_cycles": mult_specs["latency_cycles"] + add_specs["latency_cycles"],
            "energy_aJ": energy_aJ,
            "energy_J": energy_aJ * 1e-18
        }
