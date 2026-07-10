"""
simulator.py: Nanomagnetic Logic (NML) / MQCA & CMOS Baseline Simulation Engine.
Upgraded with physical physics models: thermodynamics, switching errors, and area.
"""

import math

class MQCAModel:
    def __init__(self, energy_per_magnet_switch_aJ=2.0, switching_activity=0.5,
                 temp_K=300.0, magnet_volume_m3=3.0e-24, uniaxial_anisotropy_J_m3=4.0e4,
                 cell_dimension_nm=40.0):
        """
        Initializes the MQCA model with physical parameters.
        :param energy_per_magnet_switch_aJ: Switching energy per nanomagnet in attojoules (aJ = 10^-18 J).
        :param switching_activity: Average fraction of magnets that switch state per cycle (default 0.5).
        :param temp_K: Device temperature in Kelvin (default 300K).
        :param magnet_volume_m3: Physical volume of a single nanomagnet (default 3e-24 m3).
        :param uniaxial_anisotropy_J_m3: Magnetic anisotropy constant (default 4e4 J/m3).
        :param cell_dimension_nm: Dimension of an MQCA layout grid cell (default 40nm).
        """
        self.energy_per_magnet_switch_aJ = energy_per_magnet_switch_aJ
        self.switching_activity = switching_activity
        self.temp_K = temp_K
        self.magnet_volume_m3 = magnet_volume_m3
        self.uniaxial_anisotropy_J_m3 = uniaxial_anisotropy_J_m3
        self.cell_area_um2 = (cell_dimension_nm * 1e-3) ** 2  # (0.040 um)^2 = 0.0016 um^2

        # Physics constants
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
        
        # Thermodynamic Minimum (Landauer Limit): E_min = k_B * T * ln(2)
        self.thermal_min_energy_J = self.k_B * self.temp_K * math.log(2.0)
        self.thermal_min_energy_aJ = self.thermal_min_energy_J * 1e18

        # Thermal stability barrier: E_b = K_u * V
        self.thermal_barrier_J = self.uniaxial_anisotropy_J_m3 * self.magnet_volume_m3
        
        # Stability ratio (Delta = E_b / k_B * T)
        self.stability_ratio = self.thermal_barrier_J / (self.k_B * self.temp_K)
        
        # Single magnet switching error rate probability: P_err = exp(-E_b / k_B*T)
        self.magnet_error_rate = math.exp(-self.stability_ratio)

    def get_adder_specs(self, bit_width):
        """
        Calculates resources for an N-bit MQCA adder.
        """
        magnet_count = 35 * bit_width
        gate_count = 4 * bit_width
        latency_cycles = 0.5 * bit_width + 1.0
        return {
            "magnets": magnet_count,
            "gates": gate_count,
            "latency_cycles": latency_cycles
        }

    def get_multiplier_specs(self, bit_width):
        """
        Calculates resources for an N-bit MQCA multiplier.
        """
        magnet_count = 30 * (bit_width ** 2)
        gate_count = 3 * (bit_width ** 2)
        latency_cycles = 1.5 * bit_width
        return {
            "magnets": magnet_count,
            "gates": gate_count,
            "latency_cycles": latency_cycles
        }

    def get_mac_specs(self, bit_width):
        """
        Calculates resources for an N-bit MAC unit (A * B + C).
        Upgraded with error rates and layout area calculations.
        """
        mult_specs = self.get_multiplier_specs(bit_width)
        acc_bit_width = 2 * bit_width + 4
        add_specs = self.get_adder_specs(acc_bit_width)

        total_magnets = mult_specs["magnets"] + add_specs["magnets"]
        total_gates = mult_specs["gates"] + add_specs["gates"]
        total_latency = mult_specs["latency_cycles"] + add_specs["latency_cycles"]

        # Base energy calculation
        energy_aJ = total_magnets * self.switching_activity * self.energy_per_magnet_switch_aJ
        # Check against thermodynamic landauer limit for sanity
        min_adiabatic_energy_aJ = total_magnets * self.switching_activity * self.thermal_min_energy_aJ
        if energy_aJ < min_adiabatic_energy_aJ:
            energy_aJ = min_adiabatic_energy_aJ  # Enforce Landauer physics limit
        energy_J = energy_aJ * 1e-18

        # Layout Area estimation (um2)
        area_um2 = total_magnets * self.cell_area_um2

        # Block-level switching error rate: 1 - (1 - P_err)^M
        mac_error_rate = 1.0 - ((1.0 - self.magnet_error_rate) ** total_magnets)

        return {
            "magnets": total_magnets,
            "gates": total_gates,
            "latency_cycles": total_latency,
            "energy_aJ": energy_aJ,
            "energy_J": energy_J,
            "area_um2": area_um2,
            "error_rate": mac_error_rate
        }


class CMOSModel:
    def __init__(self, process_node_nm=28):
        self.process_node_nm = process_node_nm

    def get_mac_energy_J(self, bit_width):
        """
        Returns estimated energy for standard 28nm CMOS MAC operations in Joules.
        """
        if bit_width <= 4:
            energy_fJ = 35.0
        elif bit_width <= 8:
            energy_fJ = 150.0
        elif bit_width <= 16:
            energy_fJ = 400.0
        else:
            energy_fJ = 3700.0
        return energy_fJ * 1e-15

    def get_mac_area_um2(self, bit_width):
        """
        Returns estimated silicon area for a 28nm CMOS MAC unit.
        - INT4 MAC: ~25 um2
        - INT8 MAC: ~100 um2
        - FP16 MAC: ~350 um2
        """
        if bit_width <= 4:
            return 25.0
        elif bit_width <= 8:
            return 100.0
        elif bit_width <= 16:
            return 350.0
        else:
            return 2500.0


def run_self_check():
    mqca = MQCAModel()
    cmos = CMOSModel()

    print("=== MQCA Physics Upgraded Model ===")
    print(f"Landauer Minimum Limit per switch: {mqca.thermal_min_energy_aJ:.5f} aJ")
    print(f"Thermal Stability (Eb/kBT): {mqca.stability_ratio:.1f}")
    print(f"Magnet Error Rate: {mqca.magnet_error_rate:.2e}")

    for bw in [4, 8]:
        specs = mqca.get_mac_specs(bw)
        cmos_energy = cmos.get_mac_energy_J(bw)
        cmos_area = cmos.get_mac_area_um2(bw)
        print(f"\n--- INT{bw} MAC Comparison ---")
        print(f"MQCA: Energy={specs['energy_aJ']:.1f} aJ | Area={specs['area_um2']:.5f} um2 | Block Error={specs['error_rate']:.2e}")
        print(f"CMOS: Energy={cmos_energy*1e15:.1f} fJ | Area={cmos_area:.1f} um2")
        print(f"-> Energy savings: {cmos_energy / specs['energy_J']:.1f}x")
        print(f"-> Area density increase: {cmos_area / specs['area_um2']:.1f}x")

if __name__ == "__main__":
    run_self_check()
