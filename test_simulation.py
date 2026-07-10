import pytest
import math
from simulator import MQCAModel, CMOSModel
from mapper import WorkloadMapper

def test_workload_mapper_calculation():
    mapper = WorkloadMapper(sram_cache_size_bytes=2000, sram_energy_pJ_per_byte=1.0)
    profiled_layers = [
        {"layer_id": "l.hit", "model": "T", "type": "Conv2d", "macs": 1000, "total_memory_bytes": 1000, "arithmetic_intensity": 1.0},
        {"layer_id": "l.miss", "model": "T", "type": "Conv2d", "macs": 1000, "total_memory_bytes": 4000, "arithmetic_intensity": 1.0}
    ]
    res = mapper.map_workload(profiled_layers)
    assert res[0]["cache_status"] == "HIT (SRAM)"
    assert res[1]["cache_status"] == "MISS (DRAM)"
