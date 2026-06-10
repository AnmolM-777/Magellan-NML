class WorkloadProfiler:
    def __init__(self):
        pass
    def get_fallback_profile(self):
        return [
            {'layer_id': 'midas.stem.conv', 'model': 'MiDaS', 'type': 'Conv2d', 'macs': 150000000, 'weights_count': 3120, 'input_shape': (3, 256, 256), 'output_shape': (32, 128, 128)},
            {'layer_id': 'yolo.stem.conv', 'model': 'YOLOv8-nano', 'type': 'Conv2d', 'macs': 44236800, 'weights_count': 432, 'input_shape': (3, 640, 640), 'output_shape': (16, 320, 320)}
        ]
