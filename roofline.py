class RooflineAnalyzer:
    def __init__(self, memory_bandwidth_GBs=51.2):
        self.bandwidth_Bs = memory_bandwidth_GBs * 1e9

    def classify(self, ai, ceiling_ops):
        ridge = ceiling_ops / self.bandwidth_Bs
        if ai < ridge:
            return 'Memory-Bound'
        return 'Compute-Bound'

    def run_sensitivity(self, results, base_ceil, target_ceil):
        table = []
        for r in results:
            base = self.classify(r['arithmetic_intensity'], base_ceil)
            target = self.classify(r['arithmetic_intensity'], target_ceil)
            table.append({'layer_id': r['layer_id'], 'base': base, 'target': target})
        return table
