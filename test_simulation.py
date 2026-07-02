import pytest
from simulator import MQCAModel, CMOSModel

def test_correctness():
    mqca = MQCAModel()
    assert mqca.get_adder_specs(8)['magnets'] == 280
