"""
simulator.py: Nanomagnetic Logic (NML) / MQCA & CMOS Baseline Simulation Engine.
Models arithmetic primitives and scales energy and latency metrics based on bit-width.
"""
import math

class MQCAModel:
    def __init__(self, energy_per_magnet_switch_aJ=2.0, switching_activity=0.5, temp_K=300.0,
                 magnet_volume_m3=3.0e-24, uniaxial_anisotropy_J_m3=4.0e4):
        self.energy_per_magnet_switch_aJ = energy_per_magnet_switch_aJ
        self.switching_activity = switching_activity
        self.temp_K = temp_K
        self.magnet_volume_m3 = magnet_volume_m3
        self.uniaxial_anisotropy_J_m3 = uniaxial_anisotropy_J_m3
        
        self.k_B = 1.380649e-23
        self.thermal_min_energy_J = self.k_B * self.temp_K * math.log(2.0)
        self.thermal_min_energy_aJ = self.thermal_min_energy_J * 1e18

        # Stability barrier: Eb = Ku * V
        self.thermal_barrier_J = self.uniaxial_anisotropy_J_m3 * self.magnet_volume_m3
        self.stability_ratio = self.thermal_barrier_J / (self.k_B * self.temp_K)
        # Single magnet error rate
        self.magnet_error_rate = math.exp(-self.stability_ratio)

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
        
        min_adiabatic_energy_aJ = total_magnets * self.switching_activity * self.thermal_min_energy_aJ
        if energy_aJ < min_adiabatic_energy_aJ:
            energy_aJ = min_adiabatic_energy_aJ
            
        mac_error_rate = 1.0 - ((1.0 - self.magnet_error_rate) ** total_magnets)
        
        return {
            "magnets": total_magnets,
            "gates": mult_specs["gates"] + add_specs["gates"],
            "latency_cycles": mult_specs["latency_cycles"] + add_specs["latency_cycles"],
            "energy_aJ": energy_aJ,
            "energy_J": energy_aJ * 1e-18,
            "error_rate": mac_error_rate
        }
