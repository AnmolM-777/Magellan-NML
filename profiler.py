class WorkloadProfiler:
    def __init__(self):
        pass
    def get_fallback_profile(self):
        return [
            {'layer_id': 'midas.stem.conv', 'model': 'MiDaS', 'type': 'Conv2d', 'macs': 150000000, 'weights_count': 3120, 'input_shape': (3, 256, 256), 'output_shape': (32, 128, 128)},
            {'layer_id': 'yolo.stem.conv', 'model': 'YOLOv8-nano', 'type': 'Conv2d', 'macs': 44236800, 'weights_count': 432, 'input_shape': (3, 640, 640), 'output_shape': (16, 320, 320)}
        ]
    def run_profiler(self, bit_width=8):
        profile = self.get_fallback_profile()
        results = []
        for p in profile:
            bytes_factor = bit_width / 8.0
            # Simple estimation
            mem = p['weights_count'] * bytes_factor
            results.append({
                'layer_id': p['layer_id'],
                'model': p['model'],
                'type': p['type'],
                'macs': p['macs'],
                'total_memory_bytes': mem,
                'arithmetic_intensity': p['macs'] / mem if mem > 0 else 0
            })
        return results
