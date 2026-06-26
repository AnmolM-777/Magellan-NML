import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class RooflineAnalyzer:
    def __init__(self, memory_bandwidth_GBs=51.2):
        self.bandwidth_Bs = memory_bandwidth_GBs * 1e9

    def classify(self, ai, ceiling_ops):
        ridge = ceiling_ops / self.bandwidth_Bs
        if ai < ridge:
            return 'Memory-Bound'
        return 'Compute-Bound'

    def generate_plots(self, results):
        plt.figure()
        intensities = np.logspace(-2, 2, 100)
        # Plot bandwidth bound
        plt.loglog(intensities, intensities * (self.bandwidth_Bs / 1e9))
        plt.title('Roofline Model')
        plt.savefig('roofline.png')
        plt.close()
