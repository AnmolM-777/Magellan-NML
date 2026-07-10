"""
simulator.py: Nanomagnetic Logic (NML) / MQCA & CMOS Baseline Simulation Engine.
Models arithmetic primitives and scales energy and latency metrics based on bit-width.
"""

import math

class MQCAModel:
    def __init__(self, energy_per_magnet_switch_aJ=2.0, switching_activity=0.5):
        """
        Initializes the MQCA model with hardware parameters.
        :param energy_per_magnet_switch_aJ: Switching energy per nanomagnet in attojoules (aJ = 10^-18 J).
                                            Default is 2.0 aJ based on room-temperature ferromagnet parameters.
        :param switching_activity: Average fraction of magnets that switch state per cycle (default 0.5).
        """
        self.energy_per_magnet_switch_aJ = energy_per_magnet_switch_aJ
        self.switching_activity = switching_activity

    def get_adder_specs(self, bit_width):
        """
        Calculates the resources for an N-bit ripple-carry or carry-lookahead MQCA adder.
        Based on optimized NML adder literature where 1-bit full adder is ~35 magnets.
        """
        magnet_count = 35 * bit_width
        gate_count = 4 * bit_width
        # Latency in clock cycles (multiphase clock zones)
        latency_cycles = 0.5 * bit_width + 1.0
        return {
            "magnets": magnet_count,
            "gates": gate_count,
            "latency_cycles": latency_cycles
        }

    def get_multiplier_specs(self, bit_width):
        """
        Calculates resources for an N-bit MQCA multiplier.
        Calibrated against Sivasubramani et al. 2021 (Nano Express):
        For 2-bit multiplier: ~120 magnets, 12 gates, 3.0 clock cycles.
        Scales quadratically with bit-width.
        """
        # Fit to 2-bit baseline: M(2)=120 -> coeff=30
        magnet_count = 30 * (bit_width ** 2)
        # Fit to 2-bit baseline: G(2)=12 -> coeff=3
        gate_count = 3 * (bit_width ** 2)
        # Latency cycles scales linearly with bit-width: L(2)=3.0 -> coeff=1.5
        latency_cycles = 1.5 * bit_width
        return {
            "magnets": magnet_count,
            "gates": gate_count,
            "latency_cycles": latency_cycles
        }

    def get_mac_specs(self, bit_width):
        """
        Calculates resources for an N-bit MAC unit (A * B + C).
        Accumulator C has bit-width (2 * bit_width + 4) to prevent overflow.
        """
        mult_specs = self.get_multiplier_specs(bit_width)
        acc_bit_width = 2 * bit_width + 4
        add_specs = self.get_adder_specs(acc_bit_width)

        total_magnets = mult_specs["magnets"] + add_specs["magnets"]
        total_gates = mult_specs["gates"] + add_specs["gates"]
        total_latency = mult_specs["latency_cycles"] + add_specs["latency_cycles"]

        # Calculate energy consumption per MAC operation in Joules
        # E = magnet_count * switching_activity * energy_per_switch
        energy_aJ = total_magnets * self.switching_activity * self.energy_per_magnet_switch_aJ
        energy_J = energy_aJ * 1e-18

        return {
            "magnets": total_magnets,
            "gates": total_gates,
            "latency_cycles": total_latency,
            "energy_aJ": energy_aJ,
            "energy_J": energy_J
        }


class CMOSModel:
    def __init__(self, process_node_nm=28):
        """
        Initializes standard CMOS energy reference values.
        Node reference: 28nm CMOS technology.
        """
        self.process_node_nm = process_node_nm

    def get_mac_energy_J(self, bit_width):
        """
        Returns estimated energy for standard CMOS MAC operations at 28nm.
        Values calibrated from Mark Horowitz's energy scaling data:
        - INT4 MAC: ~35 fJ (3.5e-14 J)
        - INT8 MAC: ~150 fJ (1.5e-13 J)
        - FP16 MAC: ~400 fJ (4.0e-13 J)
        - FP32 MAC: ~3.7 pJ (3.7e-12 J)
        """
        if bit_width <= 4:
            # 4-bit integer
            energy_fJ = 35.0
        elif bit_width <= 8:
            # 8-bit integer
            energy_fJ = 150.0
        elif bit_width <= 16:
            # 16-bit integer / FP16 proxy
            energy_fJ = 400.0
        else:
            # 32-bit integer / FP32 proxy
            energy_fJ = 3700.0

        return energy_fJ * 1e-15  # Convert femtojoules to Joules


def run_self_check():
    mqca = MQCAModel()
    cmos = CMOSModel()

    print("=== MQCA Model Specs ===")
    for bw in [4, 8]:
        specs = mqca.get_mac_specs(bw)
        cmos_energy = cmos.get_mac_energy_J(bw)
        savings = cmos_energy / specs["energy_J"]
        print(f"INT{bw} MAC MQCA: Magnets={specs['magnets']}, Latency={specs['latency_cycles']} cycles, Energy={specs['energy_aJ']:.1f} aJ ({specs['energy_J']*1e15:.3f} fJ)")
        print(f"INT{bw} MAC CMOS: Energy={cmos_energy*1e15:.1f} fJ")
        print(f"-> Energy savings: {savings:.1f}x reduction")

if __name__ == "__main__":
    run_self_check()
