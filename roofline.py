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

    def generate_plots(self, results, ceiling_ops=102.4e9):
        plt.figure()
        intensities = np.logspace(-2, 2, 100)
        ceil_gmacs = ceiling_ops / 1e9
        plt.loglog(intensities, np.minimum(ceil_gmacs, intensities * (self.bandwidth_Bs / 1e9)))
        plt.title('Roofline Model')
        plt.savefig('roofline.png')
        plt.close()
