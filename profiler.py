class WorkloadProfiler:
    def __init__(self, use_live_profiling=False):
        self.use_live_profiling = use_live_profiling
    def estimate_quantization_accuracy(self, model, bit_width):
        if model == "YOLOv8-nano":
            if bit_width >= 8: return {"metric_name": "mAP50-95", "value": 0.365, "quality": "Very Good"}
            return {"metric_name": "mAP50-95", "value": 0.282, "quality": "Degraded"}
        else:
            if bit_width >= 8: return {"metric_name": "RMSE (meters)", "value": 0.091, "quality": "Good"}
            return {"metric_name": "RMSE (meters)", "value": 0.165, "quality": "Poor"}
