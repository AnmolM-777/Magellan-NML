from simulator import MQCAModel, CMOSModel
from profiler import WorkloadProfiler

def test_multiplier():
    model = MQCAModel()
    specs = model.get_multiplier_specs(2)
    assert specs['magnets'] == 120

def test_cmos():
    cmos = CMOSModel()
    assert cmos.get_mac_energy_J(8) == 1.5e-13

def test_profiler_footprint():
    prof = WorkloadProfiler()
    layers = prof.run_profiler(bit_width=8)
    # Check that memory bytes are correctly populated
    assert layers[0]['total_memory_bytes'] > 0
