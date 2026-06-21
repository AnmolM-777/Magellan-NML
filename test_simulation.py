from simulator import MQCAModel, CMOSModel
from profiler import WorkloadProfiler
from mapper import WorkloadMapper

def test_multiplier():
    model = MQCAModel()
    specs = model.get_multiplier_specs(2)
    assert specs['magnets'] == 120

def test_cmos():
    cmos = CMOSModel()
    assert cmos.get_mac_energy_J(8) == 1.5e-13

def test_mapper_calculations():
    mapper = WorkloadMapper()
    layers = [{'layer_id': 'test', 'model': 'test', 'macs': 1000, 'total_memory_bytes': 100, 'arithmetic_intensity': 10}]
    res = mapper.map_workload(layers)
    assert len(res) == 1
