import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class RooflineAnalyzer:
    def __init__(self, memory_bandwidth_GBs=51.2):
        self.bandwidth_Bs = memory_bandwidth_GBs * 1e9
        self.mem_interfaces = {"LPDDR5": 42.6e9, "DDR5": 51.2e9, "HBM3": 819.2e9}
