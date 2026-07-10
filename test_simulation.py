import pytest
import math
from simulator import MQCAModel, CMOSModel

def test_mqca_baseline_calibration():
    mqca = MQCAModel()
    mult_specs = mqca.get_multiplier_specs(2)
    assert mult_specs["magnets"] == 120

def test_mqca_thermodynamics_physics():
    mqca = MQCAModel(energy_per_magnet_switch_aJ=1e-6)
    expected_limit_J = 1.380649e-23 * 300.0 * math.log(2.0)
    assert mqca.thermal_min_energy_J == pytest.approx(expected_limit_J)
