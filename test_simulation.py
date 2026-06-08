from simulator import MQCAModel, CMOSModel

def test_multiplier():
    model = MQCAModel()
    specs = model.get_multiplier_specs(2)
    assert specs['magnets'] == 120

def test_cmos():
    cmos = CMOSModel()
    assert cmos.get_mac_energy_J(8) == 1.5e-13
