from simulator import MQCAModel

def test_multiplier():
    model = MQCAModel()
    specs = model.get_multiplier_specs(2)
    assert specs['magnets'] == 120
