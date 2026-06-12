from simulator import MQCAModel, CMOSModel

class WorkloadMapper:
    def __init__(self, array_size=1024):
        self.mqca = MQCAModel()
        self.cmos = CMOSModel()
        self.array_size = array_size
