"""
test_simulation.py: Unit tests for the Magellan-NML MQCA/NML simulation logic.
Validates the simulator and mapper scaling formulas.
"""

import pytest
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
    
    assert mult_specs["magnets"] == 120, "2-bit multiplier magnet count should be 120."
    assert mult_specs["gates"] == 12, "2-bit multiplier gate count should be 12."
    assert mult_specs["latency_cycles"] == 3.0, "2-bit multiplier latency should be 3.0 clock cycles."

def test_mqca_bitwidth_scaling():
    """
    Tests that the MQCA multiplier scales quadratically with bit-width.
    """
    mqca = MQCAModel()
    specs_2bit = mqca.get_multiplier_specs(2)
    specs_4bit = mqca.get_multiplier_specs(4)
    specs_8bit = mqca.get_multiplier_specs(8)
    
    # Quadratic area scaling: M(N) = 30 * N^2
    # Ratio M(4)/M(2) = 480/120 = 4x
    # Ratio M(8)/M(2) = 1920/120 = 16x
    assert specs_4bit["magnets"] == specs_2bit["magnets"] * 4
    assert specs_8bit["magnets"] == specs_2bit["magnets"] * 16

def test_cmos_reference_values():
    """
    Verifies that the CMOS reference baseline returns reasonable energies.
    """
    cmos = CMOSModel()
    # 8-bit CMOS MAC should be 150 fJ (1.5e-13 J)
    assert cmos.get_mac_energy_J(8) == pytest.approx(1.5e-13)
    # 4-bit CMOS MAC should be 35 fJ (3.5e-14 J)
    assert cmos.get_mac_energy_J(4) == pytest.approx(3.5e-14)

def test_workload_mapper_calculation():
    """
    Checks that the mapper computes comparative metrics correctly.
    """
    mapper = WorkloadMapper(nml_array_size=1024, mem_energy_pJ_per_byte=20.0)
    
    # 1 layer with 1,024,000 MACs and 1,000,000 bytes memory footprint
    profiled_layers = [{
        "layer_id": "test.layer",
        "model": "TestModel",
        "type": "Conv2d",
        "macs": 1_024_000,
        "total_memory_bytes": 1_000_000,
        "arithmetic_intensity": 1.024
    }]
    
    results = mapper.map_workload(profiled_layers, bit_width=8)
    assert len(results) == 1
    r = results[0]
    
    # Verify memory energy: 1,000,000 Bytes * 20 pJ/Byte = 20 uJ (2e-5 J)
    assert r["mem_energy_J"] == pytest.approx(2e-5)
    
    # Verify total energy = compute + memory
    # CMOS compute energy = 1,024,000 * 150 fJ = 1.536e-7 J
    # CMOS total = 1.536e-7 + 2e-5 J = 2.01536e-5 J
    assert r["cmos_total_energy_J"] == pytest.approx(1.536e-7 + 2e-5)
