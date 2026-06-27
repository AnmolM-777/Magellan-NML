import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class RooflineAnalyzer:
    def __init__(self, memory_bandwidth_GBs=51.2):
        self.bandwidth_Bs = memory_bandwidth_GBs * 1e9

    def generate_energy_plot(self, results):
        plt.figure()
        nml = [x['nml_total_energy_J'] for x in results]
        cmos = [x['cmos_total_energy_J'] for x in results]
        plt.bar(range(len(nml)), nml)
        plt.savefig('energy.png')
        plt.close()
