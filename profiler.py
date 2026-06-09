class WorkloadProfiler:
    def __init__(self):
        self.models = ['MiDaS', 'YOLOv8-nano']
    def get_midas_layers(self):
        return [{'layer_id': 'midas.stem.conv', 'macs': 150000000}]
