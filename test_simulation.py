import pytest
from roofline import RooflineAnalyzer

def test_classification():
    analyzer = RooflineAnalyzer(memory_bandwidth_GBs=50.0)
    assert analyzer.classify(1.0, 100e9) == 'Memory-Bound'
