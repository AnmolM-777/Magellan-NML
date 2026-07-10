"""
test_simulation.py: Unit tests for the Magellan-NML MQCA/NML simulation logic.
Validates the simulator, physics models, and mapper scaling formulas.
"""

import pytest
import math
from simulator import MQCAModel, CMOSModel
from mapper import WorkloadMapper

def test_mqca_baseline_calibration():
    """
    Validates that the MQCA multiplier model is calibrated correctly
    against Prof. Sivasubramani's Nano Express 2021 paper:
    - 2-bit multiplier: 120 magnets, 12 gates, 3.0 clock cycles.
    """
    mqca = MQCAModel()
    mult_specs = mqca.get_multiplier_specs(2)
    
    assert mult_specs["magnets"] == 120
    assert mult_specs["gates"] == 12
    assert mult_specs["latency_cycles"] == 3.0

def test_mqca_bitwidth_scaling():
    """
    Tests that the MQCA multiplier scales quadratically with bit-width.
    """
    mqca = MQCAModel()
    specs_2bit = mqca.get_multiplier_specs(2)
    specs_4bit = mqca.get_multiplier_specs(4)
    specs_8bit = mqca.get_multiplier_specs(8)
    
    assert specs_4bit["magnets"] == specs_2bit["magnets"] * 4
    assert specs_8bit["magnets"] == specs_2bit["magnets"] * 16

def test_cmos_reference_values():
    """
    Verifies that the CMOS reference baseline returns reasonable energies and area.
    """
    cmos = CMOSModel()
    assert cmos.get_mac_energy_J(8) == pytest.approx(1.5e-13)
    assert cmos.get_mac_energy_J(4) == pytest.approx(3.5e-14)
    assert cmos.get_mac_area_um2(8) == 100.0

def test_workload_mapper_calculation():
    """
    Checks that the mapper computes cache hit/spill memory energy correctly.
    """
    # 2MB SRAM cache
    mapper = WorkloadMapper(
        nml_array_size=1024,
        mem_energy_pJ_per_byte=20.0,
        sram_cache_size_bytes=2000,  # Tiny cache size to trigger hits/spills easily
        sram_energy_pJ_per_byte=1.0
    )
    
    # Layer 1: fits in SRAM Cache (1500 bytes <= 2000 bytes) -> HIT (1 pJ/Byte)
    # Layer 2: spills to DRAM (3000 bytes > 2000 bytes) -> MISS (20 pJ/Byte)
    profiled_layers = [
        {"layer_id": "layer.hit", "model": "Test", "type": "Conv2d", "macs": 1000, "total_memory_bytes": 1500, "arithmetic_intensity": 1.0},
        {"layer_id": "layer.miss", "model": "Test", "type": "Conv2d", "macs": 1000, "total_memory_bytes": 3000, "arithmetic_intensity": 1.0}
    ]
    
    results = mapper.map_workload(profiled_layers, bit_width=8)
    
    # Layer 1 memory energy = 1500 B * 1.0 pJ/B = 1500 pJ = 1.5e-9 J
    assert results[0]["cache_status"] == "HIT (SRAM)"
    assert results[0]["mem_energy_J"] == pytest.approx(1.5e-9)
    
    # Layer 2 memory energy = 3000 B * 20.0 pJ/B = 60000 pJ = 6.0e-8 J
    assert results[1]["cache_status"] == "MISS (DRAM)"
    assert results[1]["mem_energy_J"] == pytest.approx(6.0e-8)

def test_mqca_thermodynamics_physics():
    """
    Validates physical parameters: Landauer limit energy floor, stability barriers,
    and area footprint scaling.
    """
    # Initialize with default 300K
    mqca = MQCAModel(energy_per_magnet_switch_aJ=1e-6)  # Tiny baseline switch energy to force Landauer floor
    
    # Landauer limit at 300K: k_B * T * ln(2) = 1.38e-23 * 300 * 0.693 = 2.87e-21 J = 0.00287 aJ
    expected_limit_J = 1.380649e-23 * 300.0 * math.log(2.0)
    assert mqca.thermal_min_energy_J == pytest.approx(expected_limit_J)
    
    specs = mqca.get_mac_specs(8)
    
    # Verify energy is capped at the Landauer limit floor rather than going below it
    expected_floor_J = specs["magnets"] * mqca.switching_activity * expected_limit_J
    assert specs["energy_J"] == pytest.approx(expected_floor_J)

    # Verify physical layout area
    # Area = total_magnets * cell_area = 2620 * 0.0016 um2 = 4.192 um2
    # Mac magnets = 1920 (multiplier) + 35 * 20 (adder) = 2620
    assert specs["magnets"] == 2620
    assert specs["area_um2"] == pytest.approx(2620 * 0.0016)

    # Verify switching block error probability is bounded [0, 1]
    assert 0.0 < specs["error_rate"] < 1.0
