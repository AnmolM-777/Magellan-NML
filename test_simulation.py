from roofline import RooflineAnalyzer

def test_classification():
    analyzer = RooflineAnalyzer(memory_bandwidth_GBs=50.0)
    # Ceiling 100 Gops. Ridge = 100G / 50G = 2 MAC/Byte
    assert analyzer.classify(1.0, 100e9) == 'Memory-Bound'
    assert analyzer.classify(3.0, 100e9) == 'Compute-Bound'
