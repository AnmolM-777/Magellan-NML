import pytest
from simulator import CMOSModel

def test_cmos_approx():
    cmos = CMOSModel()
    assert cmos.get_mac_energy_J(8) == pytest.approx(1.5e-13)
